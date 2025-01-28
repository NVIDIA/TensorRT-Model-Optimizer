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

"""Plugin to add Medusa support for megatron-core GPT model."""

import copy
import warnings
from typing import Optional

import megatron.core
import torch
import torch.nn.functional as F
from megatron.core import InferenceParams, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import get_data_parallel_rank, get_tensor_model_parallel_rank
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint
from packaging.version import Version

from ..eagle.conversion import EagleDMRegistry
from ..eagle.eagle_model import EagleModel
from ..medusa.conversion import MedusaDMRegistry
from ..medusa.medusa_model import MedusaModel


def mcore_version_higher_than(target_version: str):
    """Check if megatron-core is least this version."""
    return Version(megatron.core.__version__) > Version(target_version)


class MedusaLayer(MegatronModule):
    """MedusaLayer impl following TensorRT-LLM's model definition.

    Medusa layer consists of a column parallel linear following a silu.
    """

    def __init__(self, config):
        """Constructor.

        Args:
            config: MCore transformer config
        """
        super().__init__(config=config)

        self.activation_func = F.silu

        self.linear = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            config.hidden_size,
            config=config,
            init_method=config.init_method,
            bias=True,
            skip_bias_add=False,
            gather_output=True,
            skip_weight_param_allocation=False,
        )

    def forward(self, x):
        """Forward function."""
        y, _ = self.linear(x)
        return x + self.activation_func(y), None

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Return MCore sharded_state_dict."""
        return self.linear.sharded_state_dict(f"{prefix}linear.", sharded_offsets, metadata)


class MedusaHead(MegatronModule):
    """MedusaHead impl following TensorRT-LLM's model definition.

    https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/models/medusa/model.py
    Medusa head consists of several MedusaLayers and an lm_head.
    """

    def __init__(self, config, vocab_size: int, num_layers: int = 1, parallel_output: bool = True):
        """Constructor.

        Args:
            config: MCore transformer config
            vocab_size: vocabulary size
            num_layers: number of Medusa layers
            parallel_output: if False, then all_gather the logits
        """
        super().__init__(config=config)

        self.medusa_layers = torch.nn.ModuleList([MedusaLayer(config) for _ in range(num_layers)])

        self.lm_head = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            vocab_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=not parallel_output,
            skip_weight_param_allocation=False,
        )

        def load_state_dict_post_hook(module, incompatible_keys):
            incompatible_keys.missing_keys.clear()
            incompatible_keys.unexpected_keys.clear()

        self.register_load_state_dict_post_hook(load_state_dict_post_hook)

    def forward(self, x):
        """Forward function."""
        for layer in self.medusa_layers:
            x, _ = layer(x)
        return self.lm_head(x)

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict = None
    ) -> ShardedStateDict:
        """Return MCore sharded_state_dict."""
        assert not sharded_offsets, "Unexpected sharded offsets"
        sharded_state_dict = {}
        layer_prefix = f"{prefix}medusa_layers."
        for i, layer in enumerate(self.medusa_layers):
            state_dict_prefix = f"{layer_prefix}{i}."
            sharded_pp_offset = []
            layer_sharded_state_dict = layer.sharded_state_dict(
                state_dict_prefix, sharded_pp_offset, metadata
            )
            sharded_state_dict.update(layer_sharded_state_dict)
        sharded_state_dict.update(
            self.lm_head.sharded_state_dict(f"{prefix}lm_head.", sharded_offsets, metadata)
        )
        return sharded_state_dict


@MedusaDMRegistry.register({GPTModel: "megatron.core.models.gpt.GPTModel"})
class _DynamicMedusaGPTModel(MedusaModel):
    """A ``megatron.core.models.gpt.GPTModel`` model with dynamic hyperparams."""

    def _setup(self):
        super()._setup()
        self._register_temp_attribute("medusa_report_acc", True)
        self._register_temp_attribute("medusa_freeze_base_model", True)
        self._register_temp_attribute("calibration_mode", False)

    def modify(
        self,
        medusa_num_heads=0,
        medusa_num_layers=0,
        medusa_freeze_base_model=True,
        medusa_report_acc=True,
    ):
        """Constructor.

        Args:
            config: MedusaConfig that specifies the medusa head configuration as well as
                    weights of base model and medusa head.
        """
        if self.config.pipeline_model_parallel_size > 1:
            warnings.warn(
                "Pipeline parallelism detected! _DynamicMedusaGPTModel only supports "
                "pipeline parallelism during TensorRT-LLM checkpoint export."
            )
        super().modify(medusa_num_heads=medusa_num_heads, medusa_num_layers=medusa_num_layers)

        self.medusa_report_acc = medusa_report_acc
        self.medusa_freeze_base_model = medusa_freeze_base_model

        # Freeze all parameters
        if self.medusa_freeze_base_model:
            for name, param in self.named_parameters():
                param.requires_grad = False

        if self.post_process:
            self.medusa_heads = torch.nn.ModuleList(
                [
                    MedusaHead(self.config, self.vocab_size, num_layers=self.medusa_num_layers)
                    for _ in range(self.medusa_num_heads)
                ]
            )

    def forward(self, *args, labels: torch.Tensor = None, **kwargs):
        """Forward pass of the Medusa GPTModel.

        Returns:
            torch.Tensor: If labels are provided, then return lm_loss of all heads. Otherwise,
                return the original logits.
        """
        if self.post_process:
            # Set the post_process to False such that the forward will return the hidden_state.
            self.post_process = False
            # Calling parent's forward to get hidden_states
            hidden_states = GPTModel.forward(self, *args, labels=labels, **kwargs)
            # Reset the post_process to True
            self.post_process = True
        else:
            # return GPTModel.forward(self, *args, labels=labels, **kwargs)
            return GPTModel.forward(self, *args, labels=None, **kwargs)

        # Original output logits
        logits, _ = self.output_layer(hidden_states)

        report_acc = self.medusa_report_acc and labels is not None

        acc = []

        # Medusa heads forward. We want to run through all the heads just to make sure all modules
        # are exercised during calibration.
        for i, head in enumerate(self.medusa_heads):
            new_logits, _ = head(hidden_states)

            # If label is not provided, then this is the inference/generation case. We didn't
            # implement fast decoding; hence we want to return the original logits untouched.
            if labels is not None:
                logits = torch.cat((logits, new_logits), dim=0)

            if report_acc:
                seq_len = new_logits.shape[0]
                gathered_logits = gather_from_tensor_model_parallel_region(new_logits)
                medusa_top1 = gathered_logits.transpose(0, 1).argmax(dim=-1)
                medusa_labels = labels[:, (i + 1) * seq_len : (i + 2) * seq_len]
                top1_p = torch.eq(medusa_labels, medusa_top1).sum() / medusa_top1.numel()
                acc.append(top1_p)

        # Return the original logits untouched.
        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        if report_acc and get_tensor_model_parallel_rank() == 0:
            print("Medusa Training Accuracy: {}".format(acc))

        loss = self.compute_language_model_loss(labels, logits)

        return loss

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict = None
    ) -> ShardedStateDict:
        """Override the shared_state_dict to take care medusa_heads."""
        assert not sharded_offsets, "Unexpected sharded offsets"

        sharded_state_dict = GPTModel.sharded_state_dict(self, prefix, sharded_offsets, metadata)

        if not hasattr(self, "medusa_heads") or self.medusa_heads is None:
            return sharded_state_dict

        # This is a remedy for nn.ModuleList. GPTModel.sharded_state_dict() is calling into
        # MegatronModule.sharded_state_dict() which requires all children to implement
        # sharded_state_dict(). medusa_heads is an nn.ModuleList which only has state_dict()
        # implemented. As a result, all the submodules will not be sharded.
        #
        # The remedy is to pop all medusa_heads* out and call the MedusaHead sharded_state_dict()
        # again to populate the correct sharded_staet_dict.
        extra_keys = []
        for key, val in sharded_state_dict.items():
            if "medusa_heads" in key:
                extra_keys += [key]
        for key in extra_keys:
            sharded_state_dict.pop(key, None)

        layer_prefix = f"{prefix}medusa_heads."
        for i, layer in enumerate(self.medusa_heads):
            layer_sharded_state_dict = layer.sharded_state_dict(f"{layer_prefix}{i}.", [], metadata)
            sharded_state_dict.update(layer_sharded_state_dict)
        return sharded_state_dict


class EagleLanguageModelEmbedding(LanguageModelEmbedding):
    """Allow last pp stage to also load the embedding."""

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple[tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
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
                replica_id=(0, 1, get_data_parallel_rank(with_context_parallel=True)),
            )
        }


class EagleTransformerBlock(TransformerBlock):
    """Only store the EAGLE decoder in the last pp stage."""

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict = None
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

    def __init__(self, config, num_layers: int, transformer_layer_spec, use_last_layernorm):
        """Constructor.

        EagleModule is essentially a GPTModel except that it only exists in
        the last pp stage. As a result, pre_process must be True (otherwise
        the decoder expects the input is from the receive buffer).
        post_process must be True to perform the final_layernorm.

        Args:
            config: MCore transformer config
            num_layers: number of Eagle layers
            transformer_layer_spec: Megatron core mode spec
        """
        super().__init__(config=config)

        eagle_config = copy.deepcopy(config)
        eagle_config.num_layers = num_layers
        eagle_config.pipeline_model_parallel_size = 1
        self.fc = tensor_parallel.ColumnParallelLinear(
            config.hidden_size * 2,
            config.hidden_size,
            config=eagle_config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=True,
            skip_weight_param_allocation=False,
        )
        # Eagle does not use the final_layernorm in decoder. It distills the base model
        # final_layernorm to eagle_module hidden_states directly.
        self.decoder = EagleTransformerBlock(
            config=eagle_config,
            spec=transformer_layer_spec,
            post_layer_norm=use_last_layernorm,
            pre_process=True,
            post_process=True,
        )

    def forward(
        self,
        decoder_input: torch.Tensor,
        attention_mask: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
    ) -> torch.Tensor:
        """Forward function."""
        decoder_input, _ = self.fc(decoder_input)

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
        return hidden_states


@EagleDMRegistry.register({GPTModel: "megatron.core.models.gpt.GPTModel"})
class _DynamicEagleGPTModel(EagleModel):
    """A ``megatron.core.models.gpt.GPTModel`` model with dynamic hyperparams."""

    def _setup(self):
        super()._setup()
        self._register_temp_attribute("eagle_self_logit_distillation", False)
        self._register_temp_attribute("eagle_freeze_base_model", True)
        self._register_temp_attribute("calibration_mode", False)

    def modify(
        self,
        eagle_num_layers=0,
        use_input_layernorm_in_first_layer=True,
        use_last_layernorm=False,
        eagle_self_logit_distillation=False,
        eagle_freeze_base_model=True,
        eagle_report_acc=True,
    ):
        if self.config.pipeline_model_parallel_size > 1:
            warnings.warn(
                "Pipeline parallelism detected! _DynamicEagleGPTModel only supports "
                "pipeline parallelism during TensorRT-LLM checkpoint export."
            )
        super().modify(
            eagle_num_layers=eagle_num_layers,
            use_input_layernorm_in_first_layer=use_input_layernorm_in_first_layer,
            use_last_layernorm=use_last_layernorm,
        )
        self.eagle_report_acc = eagle_report_acc
        self.eagle_self_logit_distillation = eagle_self_logit_distillation
        self.eagle_freeze_base_model = eagle_freeze_base_model

        if self.share_embeddings_and_output_weights:
            raise ValueError("For EAGLE, args.share_embeddings_and_output_weights must be False")

        if self.position_embedding_type != "rope":
            raise ValueError("For EAGLE, only rotary embedding is supported")

        if not self.pre_process and self.post_process:
            self.embedding = EagleLanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=self.position_embedding_type,
            )

        # Freeze all parameters
        if self.eagle_freeze_base_model:
            for name, param in self.named_parameters():
                param.requires_grad = False

        # Only the last PP stage has the additional projection and decoder layer.
        # This is to simplify the export.
        if self.post_process:
            self.eagle_module = EagleModule(
                self.config,
                self.eagle_num_layers,
                self.transformer_layer_spec,
                self.use_last_layernorm,
            )

        # Eagle loss functions
        self.eagle_loss = torch.nn.SmoothL1Loss(reduction="none")
        # self.kld = LogitsKLLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        **kwargs,
    ) -> torch.Tensor:
        # Word and rotary positional embeddings
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from decoder.input_tensor
            decoder_input = None

        if mcore_version_higher_than("0.9.0"):
            extra_kwargs = {"packed_seq_params": None}
        else:
            extra_kwargs = {}
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            inference_params,
            self.decoder,
            decoder_input,
            self.config,
            **extra_kwargs,
        )
        rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # Run base modeld decoder forward.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        # Typically, this is only the case when PP > 1.
        if not self.post_process:
            return hidden_states

        logits_sbh, _ = self.output_layer(hidden_states, weight=None)

        if inference_params is None or self.calibration_mode:
            # EAGLE needs to know the newly generated token logits[-1, :, :]
            logits_bsh = logits_sbh[:-1, :, :].transpose(0, 1)
            gathered_logits_bsh = gather_from_tensor_model_parallel_region(logits_bsh)
            prob_bsh = torch.softmax(gathered_logits_bsh, dim=-1)
            eager_ids = prob_bsh.argmax(dim=-1)

            eagle_embeddings = self.embedding(
                input_ids=eager_ids,
                position_ids=position_ids[:, 1:],
            )
            # cat([s - 1, b, h], [s - 1, b, h], dim-1) = [s - 1, b, 2h]
            eagle_decoder_input = torch.cat((eagle_embeddings, hidden_states[:-1, :, :]), dim=-1)
            eagle_hidden_states = self.eagle_module(
                eagle_decoder_input,
                attention_mask[:, :, 1:, 1:],  # TODO (chenhany): this may needs some fix
                rotary_pos_emb=rotary_pos_emb[1:, :, :, :],
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                **(extra_block_kwargs or {}),
            )
            eagle_logits, _ = self.output_layer(eagle_hidden_states)

        # If labels are not provided, return the original logits. We only return after
        # all eagle weights have been exercised for quantization calibration purpose.
        if labels is None:
            return logits_sbh.transpose(0, 1).contiguous()

        # Compute hidden state regression loss
        regression_loss = (
            self.eagle_loss(
                eagle_hidden_states,
                hidden_states[1:, :, :],
            )
            .mean(dim=-1)
            .transpose(0, 1)
        )

        # Compute lm loss (classification loss) or KLDivergence
        if self.eagle_self_logit_distillation:
            lm_loss = self.kld(eagle_logits, logits_sbh[1:, :, :])
        else:
            lm_loss = self.compute_language_model_loss(labels[:, 1:], eagle_logits)

        acc = []
        if self.eagle_report_acc:
            with torch.no_grad():
                gathered_logits = gather_from_tensor_model_parallel_region(eagle_logits)
                eagle_top1 = gathered_logits.transpose(0, 1).argmax(dim=-1)
                top1_p = torch.eq(labels[:, 1:], eagle_top1).sum() / eagle_top1.numel()
                acc.append(top1_p)

            if get_tensor_model_parallel_rank() == 0:
                print("EAGLE Training Accuracy: {}".format(acc))

        return lm_loss + 10.0 * regression_loss

    def tree_decode(self, input_ids: torch.Tensor, tree=None):
        self.eval()

        choices = [] if tree is None else tree.choices
        draft_len = len(choices)
        num_additional_tokens = 0 if tree is None else tree.depth

        micro_batch_size, seq_len = input_ids.size()
        att_mask_len = seq_len + 1 + draft_len

        # Prepare attention masks
        attention_mask = torch.tril(
            torch.ones(
                (1, att_mask_len, att_mask_len),
                device=input_ids.device,
            ),
        ).view(1, 1, att_mask_len, att_mask_len)
        attention_mask = attention_mask < 0.5

        if draft_len > 0:
            draft_attention_mask = torch.tensor(
                tree.attn_mask,
                device=input_ids.device,
            ).view(1, 1, draft_len, draft_len)
            attention_mask[0, 0, -draft_len:, -draft_len:] = draft_attention_mask

        # Prepare position ids and rope
        position_ids = torch.arange(
            att_mask_len,
            dtype=torch.long,
            device=input_ids.device,
        )
        rotary_pos_emb = self.rotary_pos_emb(seq_len + 1 + num_additional_tokens)

        if num_additional_tokens > 0:
            rotary_pos_emb, additional_rotary_pos_emb = torch.split(
                rotary_pos_emb,
                (seq_len + 1, num_additional_tokens),
                dim=0,
            )
            for i, c in enumerate(choices):
                position_ids[seq_len + 1 + i] = seq_len + len(c)
                rotary_pos_emb = torch.cat(
                    (rotary_pos_emb, additional_rotary_pos_emb[[len(c) - 1], :, :, :]), dim=0
                )

        position_ids = position_ids.unsqueeze(0)

        # Forward
        decoder_input = self.embedding(
            input_ids=input_ids,
            position_ids=position_ids[:, :seq_len],
        )
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask[:, :, :seq_len, :seq_len],
            inference_params=None,
            rotary_pos_emb=rotary_pos_emb[:seq_len, :, :, :],
            packed_seq_params=None,
        )
        logits, _ = self.output_layer(hidden_states)

        # [s b h] => [b s h]
        all_logprob = gather_from_tensor_model_parallel_region(logits[[-1], :, :].transpose(0, 1))
        all_logprob = torch.softmax(all_logprob, dim=-1)
        top_vals, top_ids = all_logprob[:, -1, :].topk(1, dim=-1)

        if num_additional_tokens == 0:
            return top_ids

        new_tokens = top_ids
        eagle_ids = torch.cat((input_ids[:, 1:], top_ids), dim=-1)
        eagle_rotary_pos_emb = rotary_pos_emb[1 : 1 + eagle_ids.shape[-1], :, :, :]
        eagle_hidden_states = hidden_states

        for i in range(num_additional_tokens):
            eagle_position_ids = position_ids[:, 1 : 1 + eagle_ids.shape[-1]]
            eagle_attn_mask = attention_mask[
                :, :, 1 : 1 + eagle_ids.shape[-1], 1 : 1 + eagle_ids.shape[-1]
            ]
            eagle_rotary_pos_emb = rotary_pos_emb[1 : 1 + eagle_ids.shape[-1], :, :, :]

            eagle_decoder_input = self.embedding(
                input_ids=eagle_ids,
                position_ids=eagle_position_ids,
            )
            eagle_decoder_input, _ = self.eagle_linear_proj(
                torch.cat((eagle_decoder_input, eagle_hidden_states), dim=-1)
            )

            new_hidden_states = self.eagle_decoder(
                hidden_states=eagle_decoder_input,
                attention_mask=eagle_attn_mask,
                inference_params=None,
                rotary_pos_emb=eagle_rotary_pos_emb,
                packed_seq_params=None,
            )

            eagle_logits, _ = self.output_layer(new_hidden_states)

            all_logprob = gather_from_tensor_model_parallel_region(
                eagle_logits[tree.relative_ids[i], :, :].transpose(0, 1)
            )
            all_logprob = torch.softmax(all_logprob, dim=-1)

            for idx, tk in zip(tree.relative_ids[i], tree.top_k[i]):
                if tk == 0:
                    continue
                top_vals, top_ids = all_logprob[:, idx, :].topk(tk, dim=-1)

                new_tokens = torch.cat((new_tokens, top_ids), dim=-1)
                eagle_ids = torch.cat((eagle_ids, top_ids), dim=-1)

                for _ in range(tk):
                    eagle_hidden_states = torch.cat(
                        (eagle_hidden_states, new_hidden_states[[idx], :, :]), dim=0
                    )

        return new_tokens
