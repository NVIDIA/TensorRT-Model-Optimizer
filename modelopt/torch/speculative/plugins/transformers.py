# Adapted from: https://github.com/ctlllll/axolotl/blob/f86767e/src/axolotl/monkeypatch/medusa_utils.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

"""Support speculative decoding for huggingface models."""

import contextlib
import copy
from typing import Any

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import Cache, DynamicCache, PretrainedConfig, PreTrainedModel
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils import ModelOutput

from ..eagle.conversion import EagleDMRegistry
from ..eagle.eagle_model import EagleModel
from ..eagle.utils import RMSNorm, expand_mask, make_causal_mask
from ..medusa.conversion import MedusaDMRegistry
from ..medusa.medusa_model import MedusaModel
from ..utils import AcceptanceRateValidation, ResBlock

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@MedusaDMRegistry.register({PreTrainedModel: "hf.PreTrainedModel"})
class HFMedusaModel(MedusaModel):
    """Medusa Model Class for huggingface models."""

    def modify(self, medusa_num_heads=0, medusa_num_layers=0):
        """Constructor.

        Args:
            medusa_num_heads: number of medusa heads.
            medusa_num_layers: number of ResBlock layers in each head.
        """
        super().modify(medusa_num_heads=medusa_num_heads, medusa_num_layers=medusa_num_layers)
        self.config.medusa = {
            "num_medusa_heads": medusa_num_heads,
            "num_medusa_layers": medusa_num_layers,
        }

        hidden_size = self.lm_head.weight.shape[-1]
        vocab_size = self.lm_head.weight.shape[0]

        # Create a list of Medusa heads
        self.medusa_heads = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(hidden_size)] * self.medusa_num_layers),
                    nn.Linear(hidden_size, vocab_size, bias=False),
                )
                for _ in range(self.medusa_num_heads)
            ]
        )

        # Ensure medusa_head's dtype and device align with the base_model
        self.medusa_heads.to(self.lm_head.weight.dtype).to(self.lm_head.weight.device)
        self.medusa_heads.device = self.lm_head.weight.device
        if hasattr(self, "hf_device_map") and "lm_head" in self.hf_device_map:
            self.hf_device_map["medusa_heads"] = self.hf_device_map["lm_head"]

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        freeze_base_model: bool = True,
        medusa_heads_coefficient: float | None = 0.2,
        medusa_decay_coefficient: float | None = 0.8,
        **kwargs,
    ) -> Any:
        """Forward pass of the MedusaModel.

        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
        """
        # Pass input through the base model
        with torch.no_grad() if freeze_base_model else contextlib.nullcontext():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                rcache_position=cache_position,
                **kwargs,
            )
            hidden_states = outputs.last_hidden_state
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            slice_indices = (
                slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            )
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        medusa_logits = [
            self.medusa_heads[i](hidden_states[:, slice_indices, :])
            for i in range(self.medusa_num_heads)
        ]

        if labels is not None:
            loss = 0
            loss_fct = CrossEntropyLoss()
            # Base model loss
            if not freeze_base_model:
                loss_logits = logits.view(-1, logits.shape[-1])
                loss_labels = labels.view(-1)
                base_model_loss = loss_fct(loss_logits, loss_labels)
                loss += base_model_loss
            # Medusa loss
            for i in range(self.medusa_num_heads):
                labels = labels[..., 1:].contiguous()
                loss_logits = medusa_logits[i][:, : -(1 + i)].contiguous()
                loss_logits = loss_logits.view(-1, loss_logits.shape[-1])
                loss_labels = labels.view(-1)
                loss += (
                    loss_fct(loss_logits, loss_labels)
                    * medusa_decay_coefficient**i
                    * medusa_heads_coefficient
                )
        else:
            loss = None

        return ModelOutput(
            loss=loss,
            logits=logits,
            medusa_logits=medusa_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class EagleModule(nn.Module):
    """Eagle module used in EAGLE model."""

    def __init__(self, config, decoder_layer_cls, bias=False):
        """Init function for EagleModule."""
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [decoder_layer_cls(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        if config.use_last_layernorm:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Optionally, we use a smaller vocab table for eagle module
        if config.draft_vocab_size != config.vocab_size or config.has_lm_head:
            # Need an extra lm_head for eagle module since vocab size is reduced.
            assert config.draft_vocab_size <= config.vocab_size, (
                "EAGLE module's vocab size should be <= base model vocab size!"
            )

            # Initialize the buffers to zero.
            # Their values depend on specific tokenzier and calibrate dataset, and should be set in training script.
            if config.draft_vocab_size < config.vocab_size:
                self.register_buffer("d2t", torch.zeros(config.draft_vocab_size, dtype=torch.int64))
            self.eagle_lm_head = nn.Linear(
                config.hidden_size,
                config.draft_vocab_size,
                bias=False,
            )

        if not config.use_aux_hidden_state:
            # In Eagle-1, the FC concentrate input embeddings and hidden states
            self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=bias)
        else:
            # In EAGLE-3, the FC concentrate hidden states from multiple base model layers
            self.fc = nn.Linear(
                len(config.eagle_aux_hidden_state_layer_ids) * config.hidden_size,
                config.hidden_size,
                bias=bias,
            )

            first_layer_attn = self.layers[0].self_attn
            if not isinstance(first_layer_attn, LlamaAttention):
                raise ValueError("EAGLE-3 only support LlamaAttention.")

            # EAGLE-3's first attention require [input_layernorm_output, aux_hidden_states]
            first_layer_attn.register_forward_pre_hook(
                self._eagle3_attention_forward_pre_hook, with_kwargs=True
            )

            # Modify qkv projection in first layer to accept 2h hidden size.
            first_layer_attn.q_proj = nn.Linear(
                first_layer_attn.q_proj.in_features * 2,
                first_layer_attn.q_proj.out_features,
                bias=first_layer_attn.config.attention_bias,
            )
            first_layer_attn.k_proj = nn.Linear(
                first_layer_attn.k_proj.in_features * 2,
                first_layer_attn.k_proj.out_features,
                bias=first_layer_attn.config.attention_bias,
            )
            first_layer_attn.v_proj = nn.Linear(
                first_layer_attn.v_proj.in_features * 2,
                first_layer_attn.v_proj.out_features,
                bias=first_layer_attn.config.attention_bias,
            )

            # In EAGLE-3, input_embeds and hidden_states are normalized separately before concatenation.
            self.input_embeds_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

            # Disable input norm in first layer. We normed embeds and h individually before.
            self.layers[0].input_layernorm = nn.Identity()

    def _eagle3_attention_forward_pre_hook(self, module, args, kwargs):
        """Concat input_embeds and hidden_states for EAGLE-3's first attention layer."""
        if "hidden_states" not in kwargs:
            raise ValueError("hidden_states not found in kwargs")
        if self._input_embeds is None:
            raise ValueError("self._input_embeds is None")

        input_embeds = self._input_embeds
        self._input_embeds = None
        kwargs["hidden_states"] = torch.cat(
            (input_embeds, self.hidden_norm(kwargs["hidden_states"])), dim=-1
        )

        return args, kwargs

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = False,
        position_embeddings: torch.Tensor | None = None,
    ):
        """Forward function for EagleModule."""
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values.get_seq_length()
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = hidden_states.device if hidden_states is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        inputs_embeds = inputs_embeds.to(hidden_states.dtype).to(hidden_states.device)
        if self.config.use_aux_hidden_state:
            # In EAGLE-3, we save input embeddings to attribute, and use it in first decoder layer by hook function
            # Also, we normalize input embeddings and hidden states before concatenating them.
            # The default input norm in first layer attn will be disabled.
            self._input_embeds = self.input_embeds_norm(inputs_embeds)
        else:  # EAGLE-1
            hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )
            # For HF>= 4.54.0, the layer_outputs is a tensor, for older, it is a tuple.
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

        pre_norm_h = hidden_states

        post_norm_h = self.norm(hidden_states) if hasattr(self, "norm") else hidden_states

        return post_norm_h, pre_norm_h, past_key_values


@EagleDMRegistry.register({PreTrainedModel: "hf.PreTrainedModel"})
class HFEagleModel(EagleModel):
    """Eagle Model Class for huggingface models."""

    def _set_default_aux_hidden_state_layers(self):
        # Read a custom config attribute since we override num_hidden_layers for offline training
        num_layers = self.config.num_hidden_layers
        if self.eagle_offline and (num_layers is None or num_layers <= 0):
            num_layers = getattr(self.config, "num_orig_hidden_layers", 0)

        self.eagle_config.eagle_aux_hidden_state_layer_ids = [
            1,
            max(0, num_layers // 2 - 1),
            max(0, num_layers - 4),
        ]
        self.eagle_config.eagle_aux_hidden_state_layer_ids = list(
            set(self.eagle_config.eagle_aux_hidden_state_layer_ids)
        )

    def _collect_aux_hidden_states_forward_hook(self, module, input, output) -> None:
        """Collect auxiliary hidden states from base model intermediate layers, save them in attribute."""
        hidden_states = (
            output.clone().detach()
            if isinstance(output, torch.Tensor)
            else output[0].clone().detach()
        )
        self._aux_hidden_states.append(hidden_states)

    def pop_aux_hidden_states(self):
        """Return aux hidden states from base model, and clear the list."""
        # In PTQ, forward method will be called with try and except to find max batch size.
        # This leads to uncleared aux hidden states in the front of the list.
        # To fix it, we only return the last num_aux_h items in the list.
        num_aux_h = len(self.eagle_config.eagle_aux_hidden_state_layer_ids)
        aux_h_list = self._aux_hidden_states[-num_aux_h:]
        self._aux_hidden_states.clear()

        return aux_h_list

    def modify(
        self,
        eagle_offline,
        eagle_hidden_state_distillation,
        eagle_self_logit_distillation,
        eagle_freeze_base_model,
        eagle_report_acc,
        eagle_reuse_base_decoder,
        eagle_loss_decay_factor,
        eagle_architecture_config,
    ):
        """Constructor.

        Args:
            config: The config for eagle decoder layers.
        """
        super().modify(
            eagle_offline=eagle_offline,
            eagle_hidden_state_distillation=eagle_hidden_state_distillation,
            eagle_self_logit_distillation=eagle_self_logit_distillation,
            eagle_freeze_base_model=eagle_freeze_base_model,
            eagle_report_acc=eagle_report_acc,
            eagle_reuse_base_decoder=eagle_reuse_base_decoder,
            eagle_loss_decay_factor=eagle_loss_decay_factor,
            eagle_architecture_config=eagle_architecture_config,
        )
        self.eagle_config = PretrainedConfig.from_dict(eagle_architecture_config)
        self.eagle_config._attn_implementation = "sdpa"
        decoder_cls = (
            type(self.model.layers[-1]) if self.eagle_reuse_base_decoder else LlamaDecoderLayer
        )

        # Use default aux_hidden_state layers if use_aux_hidden_state is True
        # but no layer id is given
        if (
            self.eagle_config.use_aux_hidden_state
            and len(self.eagle_config.eagle_aux_hidden_state_layer_ids) == 0
        ):
            self._set_default_aux_hidden_state_layers()

        if self.config.hidden_size != self.eagle_config.hidden_size:
            raise ValueError(
                "EAGLE module hidden size "
                f"{self.eagle_config.hidden_size} must match base model hidden size "
                f"{self.config.hidden_size}!"
            )

        self.eagle_module = EagleModule(
            self.eagle_config,
            decoder_cls,
        )
        self.eagle_rotary_emb = LlamaRotaryEmbedding(config=self.eagle_config)

        if eagle_offline:
            # For offline training, the base model has no layers.
            # Read the device from the lm_head instead.
            device = self.lm_head.weight.device
        elif hasattr(self.model.layers[-1].self_attn, "o_proj"):
            device = self.model.layers[-1].self_attn.o_proj.weight.device
        elif hasattr(self.model.layers[-1].self_attn, "q_proj"):
            device = self.model.layers[-1].self_attn.q_proj.weight.device
        elif hasattr(self.model.layers[-1].self_attn, "qkv_proj"):
            device = self.model.layers[-1].self_attn.qkv_proj.weight.device
        self.eagle_module.to(self.dtype).to(device)

        # Make sure self.model.embed_tokens and self.lm_head are frozen
        for param in self.model.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False

        # EAGLE-3 auxiliary hidden_states
        if self.eagle_config.use_aux_hidden_state:
            self._aux_hidden_states = []
            for layer_idx, layer in enumerate(self.model.layers):
                if layer_idx in self.eagle_config.eagle_aux_hidden_state_layer_ids:
                    layer.register_forward_hook(self._collect_aux_hidden_states_forward_hook)

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        """Expand the 2-D attention mask to 4-D and apply causal mask."""
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def _get_eagle_module_inputs(
        self,
        input_ids,
        eagle_input_hidden_states,
        attention_mask,
        position_ids,
        eagle_cache,
    ):
        """Helper function to prepare eagle inputs for the 0th eagle forward pass."""
        b, seq_length, _ = eagle_input_hidden_states.shape
        past_key_values_length = eagle_cache.get_seq_length() if eagle_cache is not None else 0
        seq_length_with_past = seq_length + past_key_values_length

        # Prepare eagle_input_ids: Shift left 1 token
        zeropadding = torch.zeros(
            input_ids.shape[0], 1, dtype=input_ids.dtype, device=input_ids.device
        )
        eagle_input_ids = torch.cat((input_ids[:, 1:], zeropadding), dim=1)

        # Prepare attention_mask
        if attention_mask is not None:  # Shift left 1 token for attention_mask
            zeropadding = torch.zeros(
                attention_mask.shape[0], 1, dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask = torch.cat((attention_mask[:, 1:], zeropadding), dim=1)
        else:
            attention_mask = torch.ones(  # Initialize default attention_mask
                (b, seq_length_with_past), dtype=torch.bool, device=eagle_input_hidden_states.device
            )

        # Expand the 2-D attention mask to 4-D and apply causal mask.
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (b, seq_length), eagle_input_hidden_states, past_key_values_length
        )

        # Prepare position_ids
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=eagle_input_hidden_states.device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        return eagle_input_ids, attention_mask, position_ids

    def _concat_eagle_inputs(
        self,
        input_ids_0,
        eagle_input_hidden_states_0,
        attention_mask_0,
        position_ids_0,
        eagle_generated_hs,
    ):
        """Helper function to prepare eagle inputs for second-fourth eagle forward pass during training-time-testing.

        This is a slow version, focusing on the correctness only. TODO: optimize this.
        Parameters:
            input_ids_0: [b, seq_length], input_ids from the 0th eagle step
            base_model_hidden_states: [b, seq_length, h]
            eagle_input_hidden_states_0: [b, seq_length, h]
            attention_mask_0: [b, seq_length, seq_length], from the 0th eagle step.
            position_ids_0: [b, seq_length], from the 0th eagle step.
            eagle_generated_hs: [b, seq_length * n_steps, h], from the LAST eagle step.
        """
        b, seq_length, h = eagle_input_hidden_states_0.shape
        dtypemin = torch.finfo(attention_mask_0.dtype).min

        if eagle_generated_hs.shape[1] == seq_length:
            # This is the second step of eagle forward

            # Concat input_ids
            cat_input_ids = torch.cat((input_ids_0, input_ids_0), dim=-1)

            # Concat hidden_states
            cat_eagle_input_hidden_states = torch.cat(
                (
                    eagle_input_hidden_states_0,
                    torch.zeros(
                        (b, 1, h),
                        dtype=eagle_input_hidden_states_0.dtype,
                        device=eagle_input_hidden_states_0.device,
                    ),
                    eagle_generated_hs[:, :-1, :],
                ),
                dim=1,
            )

            # Expand attn_mask
            zero_mask = torch.ones_like(attention_mask_0).bool()
            mask_2_1 = attention_mask_0.clone().detach()
            mask_2_1[:, :, :, :-1] = mask_2_1[:, :, :, 1:]
            mask_2_2 = torch.ones_like(attention_mask_0).bool()
            for i in range(1, seq_length - 1):
                mask_2_2[:, :, i, i] = False
            cat_attention_mask = torch.cat(
                (
                    torch.cat((attention_mask_0, zero_mask), dim=-1),
                    torch.cat((mask_2_1, mask_2_2), dim=-1),
                ),
                dim=-2,
            )
            cat_attention_mask = cat_attention_mask.masked_fill(cat_attention_mask == 1, dtypemin)

            # Concat position_ids
            cat_position_ids = torch.cat((position_ids_0, position_ids_0), dim=-1)

        elif eagle_generated_hs.shape[1] == seq_length * 2:
            cat_input_ids = torch.cat((input_ids_0, input_ids_0, input_ids_0), dim=-1)
            cat_eagle_input_hidden_states = torch.cat(
                (
                    eagle_input_hidden_states_0,
                    torch.zeros(
                        (b, 1, h),
                        dtype=eagle_input_hidden_states_0.dtype,
                        device=eagle_input_hidden_states_0.device,
                    ),
                    eagle_generated_hs[:, :-1, :],
                ),
                dim=1,
            )
            zero_mask = torch.ones_like(attention_mask_0).bool()
            mask_2_1 = attention_mask_0.clone().detach()
            mask_2_1[:, :, :, :-1] = mask_2_1[:, :, :, 1:]
            mask_2_2 = torch.ones_like(attention_mask_0).bool()
            for i in range(1, seq_length - 1):
                mask_2_2[:, :, i, i] = False

            mask_3_1 = mask_2_1.clone().detach()
            mask_3_1[:, :, :, :-1] = mask_3_1[:, :, :, 1:]
            mask_3_2 = mask_2_2.clone().detach()
            mask_3_2[:, :, :, :-1] = mask_3_2[:, :, :, 1:]
            mask_3_2[:, :, 1, 0] = True
            mask_3_3 = mask_2_2.clone().detach()
            mask_3_3[:, :, 1, 1] = True
            cat_attention_mask = torch.cat(
                (
                    torch.cat((attention_mask_0, zero_mask, zero_mask), dim=-1),
                    torch.cat((mask_2_1, mask_2_2, zero_mask), dim=-1),
                    torch.cat((mask_3_1, mask_3_2, mask_3_3), dim=-1),
                ),
                dim=-2,
            )

            cat_attention_mask = cat_attention_mask.masked_fill(cat_attention_mask == 1, dtypemin)
            cat_position_ids = torch.cat((position_ids_0, position_ids_0, position_ids_0), dim=-1)

        elif eagle_generated_hs.shape[1] == seq_length * 3:
            cat_input_ids = torch.cat((input_ids_0, input_ids_0, input_ids_0, input_ids_0), dim=-1)
            cat_eagle_input_hidden_states = torch.cat(
                (
                    eagle_input_hidden_states_0,
                    torch.zeros(
                        (b, 1, h),
                        dtype=eagle_input_hidden_states_0.dtype,
                        device=eagle_input_hidden_states_0.device,
                    ),
                    eagle_generated_hs[:, :-1, :],
                ),
                dim=1,
            )
            zero_mask = torch.ones_like(attention_mask_0).bool()
            mask_2_1 = attention_mask_0.clone().detach()
            mask_2_1[:, :, :, :-1] = mask_2_1[:, :, :, 1:]
            mask_2_2 = torch.ones_like(attention_mask_0).bool()
            for i in range(1, seq_length - 1):
                mask_2_2[:, :, i, i] = False

            mask_3_1 = mask_2_1.clone().detach()
            mask_3_1[:, :, :, :-1] = mask_3_1[:, :, :, 1:]
            mask_3_2 = mask_2_2.clone().detach()
            mask_3_2[:, :, :, :-1] = mask_3_2[:, :, :, 1:]
            mask_3_2[:, :, 1, 0] = True
            mask_3_3 = mask_2_2.clone().detach()
            mask_3_3[:, :, 1, 1] = True

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

            cat_attention_mask = torch.cat(
                (
                    torch.cat((attention_mask_0, zero_mask, zero_mask, zero_mask), dim=-1),
                    torch.cat((mask_2_1, mask_2_2, zero_mask, zero_mask), dim=-1),
                    torch.cat((mask_3_1, mask_3_2, mask_3_3, zero_mask), dim=-1),
                    torch.cat((mask_4_1, mask_4_2, mask_4_3, mask_4_4), dim=-1),
                ),
                dim=-2,
            )
            cat_attention_mask = cat_attention_mask.masked_fill(cat_attention_mask == 1, dtypemin)
            cat_position_ids = torch.cat(
                (position_ids_0, position_ids_0, position_ids_0, position_ids_0), dim=-1
            )

        else:
            raise ValueError(
                f"EAGLE generated hidden states shape {eagle_generated_hs.shape} is not supported"
            )

        return cat_eagle_input_hidden_states, cat_input_ids, cat_attention_mask, cat_position_ids

    def _base_model_forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
        freeze_base_model,
        labels,
        **kwargs,
    ):
        # TODO: This function still use eagle_module. Ideally we should remove it,
        # so we can del model.eagle_module on the base model ranks to save memory.
        with torch.no_grad() if freeze_base_model else contextlib.nullcontext():
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_hidden_states=True,
                **kwargs,
            )
            past_key_values = outputs.past_key_values
            base_model_hidden_states = outputs.hidden_states[-1]
            base_model_logits = outputs.logits

            # Optionally, compute base model loss when we want to tune the base model.
            base_model_loss = None
            if not freeze_base_model and labels is not None:  # Base model loss
                loss_fct = CrossEntropyLoss()
                loss_logits = base_model_logits.view(-1, base_model_logits.shape[-1])
                labels = labels.view(-1)
                base_model_loss = loss_fct(loss_logits, labels)

        # Map the base model logits to the draft vocab
        if self.eagle_config.draft_vocab_size != self.eagle_config.vocab_size and self.training:
            assert hasattr(self.eagle_module, "d2t"), "d2t buffer not initialized"
            base_model_logits = self._map_logits_to_draft_vocab(base_model_logits)

        return base_model_hidden_states, base_model_logits, base_model_loss, past_key_values

    def _map_logits_to_draft_vocab(self, full_logits):
        reverse_mapping = (
            torch.arange(len(self.eagle_module.d2t)).to(self.eagle_module.d2t.device)
            + self.eagle_module.d2t
        )
        return full_logits[:, :, reverse_mapping]

    def _eagle_forward(
        self,
        eagle_input_hidden_states,
        inputs_embeds,
        attention_mask,
        position_ids,
        position_embeddings,
    ):
        eagle_postnorm_h, eagle_prenorm_h, eagle_cache = self.eagle_module(
            eagle_input_hidden_states,
            inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            position_embeddings=position_embeddings,
        )
        eagle_lm_head = (
            self.eagle_module.eagle_lm_head
            if hasattr(self.eagle_module, "eagle_lm_head")
            else self.lm_head
        )
        eagle_logits = eagle_lm_head(eagle_postnorm_h)

        return eagle_postnorm_h, eagle_prenorm_h, eagle_logits, eagle_cache

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int = 0,
        loss_mask: torch.Tensor | None = None,
        classification_loss_coefficient: float | None = 1,
        regression_loss_coefficient: float | None = 0,
        **kwargs,
    ) -> Any:
        """Forward pass of the EagleModel.

        Returns:
            hidden_states: The hidden state from the base model.
            logits: logits from the base model.
            eagle_hidden_states: The hidden state from eagle_module.
            eagle_logits: logits from the eagle_module.
        """
        if past_key_values is not None and hasattr(past_key_values, "eagle_cache"):
            eagle_cache = past_key_values.eagle_cache
        else:
            eagle_cache = None

        if self.training:
            assert eagle_cache is None, "eagle_cache should be None in training"
            assert past_key_values is None, "past_key_values should be None in training"

        if loss_mask is None:
            loss_mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)

        # ====First, we run base model forward====
        if "base_model_outputs" in kwargs:
            # Parse base model outputs forwarded from teacher
            base_outputs = kwargs["base_model_outputs"]
            base_model_hidden_states = base_outputs["base_model_hidden_states"]
            if "base_model_logits" in base_outputs:
                base_model_logits = base_outputs["base_model_logits"]
            else:
                base_model_logits = self.lm_head(base_model_hidden_states)
                if self.eagle_config.draft_vocab_size != self.eagle_config.vocab_size:
                    base_model_logits = self._map_logits_to_draft_vocab(base_model_logits)
            base_model_loss = None
            past_key_values = DynamicCache()  # Dummy cache

        else:
            base_model_hidden_states, base_model_logits, base_model_loss, past_key_values = (
                self._base_model_forward(
                    input_ids,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    self.eagle_freeze_base_model,
                    labels,
                    **kwargs,
                )
            )

        if not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        # ====Run eagle forward====
        eagle_loss = None
        if self.training:
            # In EAGLE-3, we have an additional FC layer to concentrate hidden states from multiple base model layers
            batch_size, seq_length, _ = base_model_hidden_states.shape
            if self.eagle_config.use_aux_hidden_state:
                if "base_model_outputs" in kwargs:
                    aux_hidden_states = kwargs["base_model_outputs"]["aux_hidden_states"]
                else:
                    aux_hidden_states = torch.cat(self.pop_aux_hidden_states(), dim=-1)
                eagle_input_hidden_states = self.eagle_module.fc(aux_hidden_states)
            else:
                eagle_input_hidden_states = base_model_hidden_states

            # Get eagle inputs for the first eagle forward pass
            eagle_input_ids, attention_mask_0, position_ids = self._get_eagle_module_inputs(
                input_ids,
                eagle_input_hidden_states,
                attention_mask,
                position_ids,
                eagle_cache,
            )
            with torch.no_grad():
                inputs_embeds = self.model.embed_tokens(eagle_input_ids)
            position_embeddings = self.eagle_rotary_emb(eagle_input_hidden_states, position_ids)

            # Then, we run eagle forward
            eagle_postnorm_h, eagle_prenorm_h, eagle_logits, eagle_cache = self._eagle_forward(
                eagle_input_hidden_states,
                inputs_embeds,
                attention_mask_0,
                position_ids,
                position_embeddings,
            )

            if not isinstance(eagle_cache, Cache):
                eagle_cache = DynamicCache.from_legacy_cache(eagle_cache)

            past_key_values.eagle_cache = eagle_cache

            # Compute loss on the eagle modules
            regression_loss, classification_loss, accuracy_0 = self._eagle_loss(
                base_model_hidden_states[:, 1:],
                base_model_logits[:, 1:],
                eagle_postnorm_h[:, :-1],
                eagle_logits[:, :-1],
                loss_mask[:, 1:],
            )
            eagle_loss = (
                regression_loss_coefficient * regression_loss
                + classification_loss_coefficient * classification_loss
            )

            # ====Perform training-time-testing with 3 extra eagle forward passes====
            # ====Second step of eagle forward====
            eagle_input_hidden_states_1, eagle_input_ids_1, attention_mask_1, position_ids_1 = (
                self._concat_eagle_inputs(
                    eagle_input_ids,
                    eagle_input_hidden_states,
                    attention_mask_0,
                    position_ids,
                    eagle_prenorm_h,
                )
            )
            with torch.no_grad():
                inputs_embeds = self.model.embed_tokens(eagle_input_ids_1)
            position_embeddings = self.eagle_rotary_emb(eagle_input_hidden_states_1, position_ids_1)
            eagle_postnorm_h, eagle_prenorm_h, eagle_logits, eagle_cache = self._eagle_forward(
                eagle_input_hidden_states_1,
                inputs_embeds,
                attention_mask_1,
                position_ids_1,
                position_embeddings,
            )

            regression_loss, classification_loss, accuracy_1 = self._eagle_loss(
                # base model predict +1 tok, while eagle predict +2
                # so we shift base model outputs compared to eagle outputs
                base_model_hidden_states[:, 1:],
                base_model_logits[:, 1:],
                eagle_postnorm_h[
                    :,
                    -seq_length:-1,
                ],
                eagle_logits[
                    :,
                    -seq_length:-1,
                ],
                # additionally, we mask the first n tok of eagle outputs at nth TTT step
                torch.cat(
                    (
                        torch.zeros(batch_size, 1, dtype=loss_mask.dtype, device=loss_mask.device),
                        loss_mask[:, 2:],
                    ),
                    dim=1,
                ),
            )
            eagle_loss += (
                regression_loss_coefficient * regression_loss
                + classification_loss_coefficient * classification_loss
            )

            # ====Third step of eagle forward====
            eagle_input_hidden_states_2, eagle_input_ids_2, attention_mask_2, position_ids_2 = (
                self._concat_eagle_inputs(
                    eagle_input_ids,
                    eagle_input_hidden_states,
                    attention_mask_0,
                    position_ids,
                    eagle_prenorm_h,
                )
            )
            with torch.no_grad():
                inputs_embeds = self.model.embed_tokens(eagle_input_ids_2)
            position_embeddings = self.eagle_rotary_emb(eagle_input_hidden_states_2, position_ids_2)
            eagle_postnorm_h, eagle_prenorm_h, eagle_logits, eagle_cache = self._eagle_forward(
                eagle_input_hidden_states_2,
                inputs_embeds,
                attention_mask_2,
                position_ids_2,
                position_embeddings,
            )

            regression_loss, classification_loss, accuracy_2 = self._eagle_loss(
                base_model_hidden_states[:, 1:],
                base_model_logits[:, 1:],
                eagle_postnorm_h[:, -seq_length:-1, :],
                eagle_logits[
                    :,
                    -seq_length:-1,
                ],
                torch.cat(
                    (
                        torch.zeros(batch_size, 2, dtype=loss_mask.dtype, device=loss_mask.device),
                        loss_mask[:, 3:],
                    ),
                    dim=1,
                ),
            )
            eagle_loss += (
                regression_loss_coefficient * regression_loss
                + classification_loss_coefficient * classification_loss
            )

            # ====Fourth step of eagle forward====
            eagle_input_hidden_states_3, eagle_input_ids_3, attention_mask_3, position_ids_3 = (
                self._concat_eagle_inputs(
                    eagle_input_ids,
                    eagle_input_hidden_states,
                    attention_mask_0,
                    position_ids,
                    eagle_prenorm_h,
                )
            )
            with torch.no_grad():
                inputs_embeds = self.model.embed_tokens(eagle_input_ids_3)
            position_embeddings = self.eagle_rotary_emb(eagle_input_hidden_states_3, position_ids_3)
            eagle_postnorm_h, _, eagle_logits, eagle_cache = self._eagle_forward(
                eagle_input_hidden_states_3,
                inputs_embeds,
                attention_mask_3,
                position_ids_3,
                position_embeddings,
            )

            regression_loss, classification_loss, accuracy_3 = self._eagle_loss(
                base_model_hidden_states[:, 1:],
                base_model_logits[:, 1:],
                eagle_postnorm_h[
                    :,
                    -seq_length:-1,
                ],
                eagle_logits[
                    :,
                    -seq_length:-1,
                ],
                torch.cat(
                    (
                        torch.zeros(batch_size, 3, dtype=loss_mask.dtype, device=loss_mask.device),
                        loss_mask[:, 4:],
                    ),
                    dim=1,
                ),
            )
            eagle_loss += (
                regression_loss_coefficient * regression_loss
                + classification_loss_coefficient * classification_loss
            )

        # Finally, we merge base model loss and eagle loss, raise error if both are None
        if base_model_loss is not None and eagle_loss is not None:
            loss = base_model_loss + eagle_loss
        elif base_model_loss is not None:
            loss = base_model_loss
        elif eagle_loss is not None:
            loss = eagle_loss
        else:
            loss = None
            assert not self.training, ValueError(
                "Both base_model_loss and eagle_loss are skipped. At least one loss must be computed."
            )

        train_acc = (accuracy_0, accuracy_1, accuracy_2, accuracy_3) if self.training else None

        return ModelOutput(
            loss=loss,
            logits=base_model_logits,
            past_key_values=past_key_values,
            hidden_states=base_model_hidden_states,
            train_acc=train_acc,
        )

    def _eagle_loss(
        self,
        base_model_hidden_states,
        base_model_logits,
        eagle_hidden_states,
        eagle_logits,
        loss_mask,
    ):
        """Function for EAGLE loss computing."""
        loss_mask = loss_mask[:, :, None]
        criterion = nn.SmoothL1Loss(reduction="none")
        classification_loss = nn.Softmax(dim=2)(base_model_logits) * nn.LogSoftmax(dim=2)(
            eagle_logits
        )
        classification_loss = -torch.sum(torch.sum(loss_mask * classification_loss, 2)) / (
            loss_mask.sum() + 1e-5
        )
        regression_loss = criterion(eagle_hidden_states, base_model_hidden_states)
        regression_loss = torch.sum(torch.mean(loss_mask * regression_loss, 2)) / (
            loss_mask.sum() + 1e-5
        )
        # Compute accuracy
        base_predict_tok = base_model_logits.clone().detach().argmax(dim=-1)
        eagle_predict_tok = eagle_logits.clone().detach().argmax(dim=-1)
        valid = loss_mask[:, :, 0].bool()
        correct = (base_predict_tok == eagle_predict_tok) & valid
        denom = valid.sum().clamp_min(1).float()
        accuracy = round(correct.sum().float().div(denom).item(), 3)

        return regression_loss, classification_loss, accuracy

    @torch.no_grad()
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
        base_model_outputs = super().forward(
            input_ids=input_ids,
            output_hidden_states=True,
        )

        base_model_hidden_states = base_model_outputs.hidden_states[-1]
        base_model_logits = base_model_outputs.logits
        base_token = base_model_logits[:, -1:, :].argmax(dim=-1).to(input_ids.device)

        # Early return
        if steps < 1:
            if hasattr(self, "_aux_hidden_states"):
                _ = self.pop_aux_hidden_states()
            return base_token, None

        eagle_ids = torch.cat((input_ids[:, 1:], base_token), dim=-1)

        if self.eagle_config.use_aux_hidden_state:
            # EAGLE-3
            # Only the first iteration input_hidden_states are from aux_hidden_state layers
            # Gather _aux_hidden_states from all devices before concatenation
            gathered_aux_hidden_states = self.pop_aux_hidden_states()
            gathered_aux_hidden_states = [
                h.to(input_ids.device) for h in gathered_aux_hidden_states
            ]
            eagle_input_hidden_states = self.eagle_module.fc(
                torch.cat(gathered_aux_hidden_states, dim=-1)
            )

        else:
            eagle_input_hidden_states = base_model_hidden_states

        draft_tokens = []
        for _ in range(steps):
            # Get eagle inputs for the first eagle forward pass
            _, eagle_attention_mask, eagle_position_ids = self._get_eagle_module_inputs(
                input_ids,
                eagle_input_hidden_states,
                None,
                None,
                None,
            )
            position_embeddings = self.eagle_rotary_emb(
                eagle_input_hidden_states, eagle_position_ids
            )

            _, eagle_prenorm_h, eagle_logits, _ = self._eagle_forward(
                eagle_input_hidden_states,
                self.model.embed_tokens(eagle_ids),
                eagle_attention_mask,
                eagle_position_ids,
                position_embeddings,
            )

            draft_token = eagle_logits[:, -1:, :].argmax(dim=-1)
            if self.eagle_config.draft_vocab_size != self.eagle_config.vocab_size:
                draft_token += self.eagle_module.d2t[draft_token]
            draft_tokens.append(draft_token)

            eagle_ids = torch.cat((eagle_ids, draft_token.to(eagle_ids.device)), dim=-1)
            eagle_input_hidden_states = torch.cat(
                (eagle_input_hidden_states, eagle_prenorm_h[:, -1:, :]), dim=1
            )

        draft_tokens = torch.cat(draft_tokens, dim=-1).to(base_token.device)

        return base_token, draft_tokens


class HFARValidation(AcceptanceRateValidation):
    """This is the subclass for HF model AR validation."""

    def get_ground_truth(self, input_ids, osl):
        """This function returns ground truth output tokens from the base model."""
        input_ids = copy.deepcopy(input_ids).to(torch.cuda.current_device())
        for _ in range(osl):
            input_id, _ = self.model.pseudo_speculative_generate(input_ids, steps=0)
            input_ids = torch.cat((input_ids, input_id.to(input_ids.device)), dim=-1)
            if input_id[0, 0] == self.end_token:
                break
        return input_ids
