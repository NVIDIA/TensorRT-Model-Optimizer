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

"""Support medusa for huggingface models."""

import contextlib
from typing import Any, Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import Cache, DynamicCache, PreTrainedModel
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils import ModelOutput

from ..eagle.conversion import EagleDMRegistry
from ..eagle.eagle_model import EagleModel
from ..eagle.utils import LlamaDecoderLayer, LlamaRMSNorm, expand_mask, make_causal_mask
from ..medusa.conversion import MedusaDMRegistry
from ..medusa.medusa_model import MedusaModel
from ..redrafter.conversion import RedrafterDMRegistry
from ..redrafter.redrafter_model import RedrafterModel
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
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        freeze_base_model: bool = True,
        medusa_heads_coefficient: Optional[float] = 0.2,
        medusa_decay_coefficient: Optional[float] = 0.8,
    ) -> Any:
        """Forward pass of the MedusaModel.

        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
                return_dict=return_dict,
            )
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)

        medusa_logits = [self.medusa_heads[i](hidden_states) for i in range(self.medusa_num_heads)]

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

        if not return_dict:
            output = (logits, medusa_logits)
            return (loss,) + output if loss is not None else output

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

    def __init__(self, eagle_config, bias=True):
        """Init function for EagleModule."""
        super().__init__()

        self.fc = nn.Linear(2 * eagle_config["hidden_size"], eagle_config["hidden_size"], bias=bias)
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(
                    index,
                    eagle_config["hidden_size"],
                    eagle_config["intermediate_size"],
                    eagle_config["rms_norm_eps"],
                    eagle_config["num_attention_heads"],
                    eagle_config["num_key_value_heads"],
                    eagle_config["max_position_embeddings"],
                    eagle_config["rope_theta"],
                    eagle_config["use_input_layernorm_in_first_layer"],
                )
                for index in range(eagle_config["num_hidden_layers"])
            ]
        )
        if eagle_config["use_last_layernorm"]:
            self.norm = LlamaRMSNorm(eagle_config["hidden_size"], eagle_config["rms_norm_eps"])

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        lm_head: nn.Module,
        attention_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
    ):
        """Forward function for EagleModule."""
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
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

        next_decoder_cache = ()
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if hasattr(self, "norm"):
            hidden_states = self.norm(hidden_states)

        logits = lm_head(hidden_states).to(hidden_states.device)

        return hidden_states, logits, next_decoder_cache

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = make_causal_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = expand_mask(
                attention_mask, torch.float32, tgt_len=input_shape[-1]
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

    def modify(self, eagle_num_layers, use_input_layernorm_in_first_layer, use_last_layernorm):
        """Constructor.

        Args:
            config: The config for eagle decoder layers.
        """
        super().modify(
            eagle_num_layers=eagle_num_layers,
            use_input_layernorm_in_first_layer=use_input_layernorm_in_first_layer,
            use_last_layernorm=use_last_layernorm,
        )

        self.config.eagle = {
            "num_hidden_layers": eagle_num_layers,
            "num_attention_heads": self.config.num_attention_heads,
            "intermediate_size": self.config.intermediate_size,
            "hidden_size": self.config.hidden_size,
            "num_key_value_heads": self.config.num_key_value_heads,
            "rms_norm_eps": self.config.rms_norm_eps,
            "max_position_embeddings": self.config.max_position_embeddings,
            "rope_theta": self.config.rope_theta,
            "use_input_layernorm_in_first_layer": use_input_layernorm_in_first_layer,
            "use_last_layernorm": use_last_layernorm,
        }
        self.eagle_module = EagleModule(self.config.eagle)

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
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        loss_mask: torch.Tensor = None,
        freeze_base_model: Optional[bool] = True,
        classification_loss_coefficient: Optional[float] = 0.1,
        regression_loss_coefficient: Optional[float] = 1.0,
    ) -> Any:
        """Forward pass of the EagleModel.

        Returns:
            hidden_states: The hidden state from the base model.
            logits: logits from the base model.
            eagle_hidden_states: The hidden state from eagle_module.
            eagle_logits: logits from the eagle_module.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
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
                return_dict=True,
                cache_position=cache_position,
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
        eagle_hidden_states, eagle_logits, eagle_cache = self.eagle_module(
            hidden_states,
            inputs_embeds,
            self.lm_head,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=eagle_cache,
            use_cache=True,
            output_attentions=output_attentions,
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

        if not return_dict:
            output = (logits, eagle_logits)
            return (loss,) + output if loss is not None else output

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


class Drafter(nn.Module):
    """Wrapper class for the drafter in Redrafter."""

    def __init__(self, redrafter_num_layers, hidden_size, vocab_size):
        """Init function for drafter."""
        super().__init__()

        self.lm_head = torch.nn.Sequential(
            *([ResBlock(2 * hidden_size) for _ in range(redrafter_num_layers)]),
            torch.nn.Linear(in_features=2 * hidden_size, out_features=vocab_size, bias=False),
        )

        self.rnn_u = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.rnn_w = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self, input_embs: torch.Tensor, cumsum_input_embs: torch.Tensor, hidden_states: torch.Tensor
    ):
        """Forward pass of the Drafter."""
        input_embs = torch.roll(input_embs, -1, dims=1)
        o = self.rnn_u(cumsum_input_embs)
        cumsum_input_embs = nn.SiLU()(o + self.rnn_w(input_embs))
        h = torch.cat((hidden_states, cumsum_input_embs), -1)
        logits = self.lm_head(h)

        return input_embs, cumsum_input_embs, logits


@RedrafterDMRegistry.register({PreTrainedModel: "hf.PreTrainedModel"})
class HFRedrafterModel(RedrafterModel):
    """Redrafter Model Class for huggingface models."""

    def modify(self, redrafter_predict_n_tokens=0, redrafter_num_layers=0):
        """Constructor.

        Args:
            redrafter_predict_n_tokens: number of tokens the drafter will predict.
            redrafter_num_layers: number of ResBlock layers in lm head.
        """
        super().modify(
            redrafter_predict_n_tokens=redrafter_predict_n_tokens,
            redrafter_num_layers=redrafter_num_layers,
        )

        hidden_size = self.lm_head.weight.shape[-1]
        vocab_size = self.lm_head.weight.shape[0]

        self.config.redrafter = {
            "num_draft_layers": redrafter_num_layers,
            "hidden_size": hidden_size,
            "exit_dim": 2 * hidden_size,
            "rnn": True,
        }

        self.drafter = Drafter(self.redrafter_predict_n_tokens, hidden_size, vocab_size)

        # Ensure drafter's dtype and device align with the base_model
        self.drafter.to(self.lm_head.weight.dtype).to(self.lm_head.weight.device)
        self.drafter.device = self.lm_head.weight.device
        if hasattr(self, "hf_device_map") and "lm_head" in self.hf_device_map:
            self.hf_device_map["drafter"] = self.hf_device_map["lm_head"]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        freeze_base_model: bool = True,
    ) -> Any:
        """Forward pass of the RedrafterModel."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
                return_dict=return_dict,
            )
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)

        redrafter_logits = []
        input_embs = self.model.embed_tokens(input_ids)
        cumsum_input_embs = torch.zeros_like(
            input_embs, dtype=input_embs.dtype, device=input_embs.device
        )

        for _ in range(self.redrafter_predict_n_tokens):
            input_embs, cumsum_input_embs, drafter_logits = self.drafter(
                input_embs, cumsum_input_embs, hidden_states
            )
            redrafter_logits.append(drafter_logits)

        if labels is not None:
            loss = 0
            loss_fct = CrossEntropyLoss()
            # Base model loss
            if not freeze_base_model:
                loss_logits = logits.view(-1, logits.shape[-1])
                loss_labels = labels.view(-1)
                base_model_loss = loss_fct(loss_logits, loss_labels)
                loss += base_model_loss
            # Drafter loss
            for i in range(self.redrafter_predict_n_tokens):
                labels = labels[..., 1:].contiguous()
                loss_logits = redrafter_logits[i][:, : -(1 + i)].contiguous()
                loss_logits = loss_logits.view(-1, loss_logits.shape[-1])
                loss_labels = labels.view(-1)
                loss += loss_fct(loss_logits, loss_labels)
        else:
            loss = None

        if not return_dict:
            output = (logits, redrafter_logits)
            return (loss,) + output if loss is not None else output

        return ModelOutput(
            loss=loss,
            logits=logits,
            redrafter_logits=redrafter_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
