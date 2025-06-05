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
from typing import Any

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import Cache, DynamicCache, PreTrainedModel
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils import ModelOutput

from ..eagle.conversion import EagleDMRegistry
from ..eagle.eagle_model import EagleModel
from ..eagle.utils import RMSNorm, expand_mask, make_causal_mask
from ..medusa.conversion import MedusaDMRegistry
from ..medusa.medusa_model import MedusaModel
from ..utils import ResBlock

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

    def __init__(self, config, decoder_layer_cls, num_layers, use_last_layernorm=False, bias=True):
        """Init function for EagleModule."""
        super().__init__()

        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=bias)
        self.layers = nn.ModuleList(
            [decoder_layer_cls(config, layer_idx) for layer_idx in range(num_layers)]
        )
        if use_last_layernorm:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        lm_head: nn.Module,
        attention_mask: torch.Tensor | None = None,
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

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        inputs_embeds = inputs_embeds.to(hidden_states.dtype).to(hidden_states.device)
        hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))

        for idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )

            hidden_states = layer_outputs[0]

        if hasattr(self, "norm"):
            hidden_states = self.norm(hidden_states)

        logits = lm_head(hidden_states).to(hidden_states.device)

        return hidden_states, logits, past_key_values

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
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


@EagleDMRegistry.register({PreTrainedModel: "hf.PreTrainedModel"})
class HFEagleModel(EagleModel):
    """Eagle Model Class for huggingface models."""

    def _set_default_aux_hidden_state_layers(self):
        num_layers = self.config.num_hidden_layers
        self.eagle_aux_hidden_state_layer_ids = [1, num_layers // 2 - 1, num_layers - 4]

    def modify(
        self,
        eagle_num_layers,
        use_input_layernorm_in_first_layer,
        use_last_layernorm,
        eagle_hidden_state_distillation,
        use_aux_hidden_state,
        eagle_aux_hidden_state_layer_ids,
        eagle_disable_moe,  # Not used in HFEagleModel
        draft_vocab_size,
        use_mtp_layernorm,
        ffn_hidden_size=0,
    ):
        """Constructor.

        Args:
            config: The config for eagle decoder layers.
        """
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
        )

        self.config.eagle = {
            "num_hidden_layers": eagle_num_layers,
            "num_attention_heads": self.config.num_attention_heads,
            "head_dim": getattr(
                self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads
            ),
            "intermediate_size": self.config.intermediate_size,
            "hidden_size": self.config.hidden_size,
            "num_key_value_heads": self.config.num_key_value_heads,
            "rms_norm_eps": self.config.rms_norm_eps,
            "max_position_embeddings": self.config.max_position_embeddings,
            "rope_theta": self.config.rope_theta,
            "use_input_layernorm_in_first_layer": use_input_layernorm_in_first_layer,
            "use_last_layernorm": use_last_layernorm,
        }
        self.eagle_module = EagleModule(
            self.config, type(self.model.layers[-1]), eagle_num_layers, use_last_layernorm
        )

        if hasattr(self.model.layers[-1].self_attn, "o_proj"):
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
        freeze_base_model: bool = True,
        classification_loss_coefficient: float | None = 0.1,
        regression_loss_coefficient: float | None = 1.0,
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

        with torch.no_grad() if freeze_base_model else contextlib.nullcontext():
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
            past_key_values = outputs.past_key_values
            if not isinstance(past_key_values, Cache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            hidden_states = outputs.hidden_states[-1]
            logits = outputs.logits

        # Shift left 1 token for eagle inputs
        zeropadding = torch.zeros(
            input_ids.shape[0], 1, dtype=input_ids.dtype, device=input_ids.device
        )
        eagle_input_ids = torch.cat((input_ids[:, 1:], zeropadding), dim=1)
        if attention_mask is not None:
            zeropadding = torch.zeros(
                attention_mask.shape[0], 1, dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask = torch.cat((attention_mask[:, 1:], zeropadding), dim=1)

        with torch.no_grad():
            inputs_embeds = self.model.embed_tokens(eagle_input_ids)

        _, seq_length, _ = hidden_states.shape
        device = hidden_states.device
        past_key_values_length = eagle_cache.get_seq_length() if eagle_cache is not None else 0
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        eagle_hidden_states, eagle_logits, eagle_cache = self.eagle_module(
            hidden_states,
            inputs_embeds,
            self.lm_head,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=eagle_cache,
            use_cache=True,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
        )
        if not isinstance(eagle_cache, Cache):
            eagle_cache = DynamicCache.from_legacy_cache(eagle_cache)
        past_key_values.eagle_cache = eagle_cache

        loss = None
        if not freeze_base_model and labels is not None:
            loss_fct = CrossEntropyLoss()
            loss_logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            base_model_loss = loss_fct(loss_logits, labels)
            loss = base_model_loss

        if loss_mask is not None:
            # Shift hidden_states and logits to align with eagle counterparts
            zeropadding = torch.zeros(
                hidden_states.shape[0],
                1,
                hidden_states.shape[2],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            hidden_states = torch.cat((hidden_states[:, 1:], zeropadding), dim=1).detach()
            zeropadding = torch.zeros(
                logits.shape[0], 1, logits.shape[2], dtype=logits.dtype, device=logits.device
            )
            base_model_logits = torch.cat((logits[:, 1:], zeropadding), dim=1).detach()

            regression_loss, classification_loss = self._eagle_loss(
                hidden_states, base_model_logits, eagle_hidden_states, eagle_logits, loss_mask
            )
            eagle_loss = (
                regression_loss_coefficient * regression_loss
                + classification_loss_coefficient * classification_loss
            )
            if loss is None:
                loss = eagle_loss
            else:
                loss += eagle_loss

        return ModelOutput(
            loss=loss,
            logits=logits,
            eagle_logits=eagle_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _eagle_loss(self, hidden_states, logits, eagle_hidden_states, eagle_logits, loss_mask):
        """Function for EAGLE loss computing."""
        loss_mask = loss_mask[:, :, None]
        criterion = nn.SmoothL1Loss(reduction="none")
        classification_loss = nn.Softmax(dim=2)(logits) * nn.LogSoftmax(dim=2)(eagle_logits)
        classification_loss = -torch.sum(torch.sum(loss_mask * classification_loss, 2)) / (
            loss_mask.sum() + 1e-5
        )
        regression_loss = criterion(eagle_hidden_states, hidden_states)
        regression_loss = torch.sum(torch.mean(loss_mask * regression_loss, 2)) / (
            loss_mask.sum() + 1e-5
        )
        return regression_loss, classification_loss
