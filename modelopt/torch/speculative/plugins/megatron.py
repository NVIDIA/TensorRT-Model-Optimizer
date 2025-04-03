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
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import get_data_parallel_rank, get_tensor_model_parallel_rank
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint
from packaging.version import Version

from ..eagle.conversion import EagleDMRegistry
from ..eagle.eagle_model import EagleModel
from ..medusa.conversion import MedusaDMRegistry
from ..medusa.medusa_model import MedusaModel
from ..mtp.conversion import MTPDMRegistry
from ..mtp.mtp_model import MTPModel
from ..utils import (
    AcceptanceRateValidation,
    get_default_attention_mask_and_position_ids,
    right_padding,
)


def mcore_version_higher_than(target_version: str):
    """Check if megatron-core is least this version."""
    return Version(megatron.core.__version__) > Version(target_version)


def logits_kld_loss(logits, gt_logits):
    """KL Divergence loss using ground truth logits."""
    gathered_logits = gather_from_tensor_model_parallel_region(logits)
    gathered_gt_logits = gather_from_tensor_model_parallel_region(gt_logits)
    loss = torch.nn.Softmax(dim=2)(gathered_gt_logits) * torch.nn.LogSoftmax(dim=2)(gathered_logits)
    loss = -torch.sum(loss, 2)
    return loss.transpose(0, 1)


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

        if config.use_cpu_initialization:
            device = torch.device("cpu")
        else:
            device = torch.cuda.current_device()

        self.activation_func = F.silu

        self.linear = torch.nn.Linear(
            config.hidden_size,
            config.hidden_size,
            dtype=config.params_dtype,
            device=device,
        )

    def forward(self, x):
        """Forward function."""
        y = self.linear(x)
        return x + self.activation_func(y), None


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

    def _base_model_forward(self, *args, labels: torch.Tensor = None, **kwargs):
        if self.post_process:
            # Set the post_process to False such that the forward will return the hidden_state.
            self.post_process = False
            # Calling parent's forward to get hidden_states
            hidden_states = GPTModel.forward(self, *args, labels=labels, **kwargs)
            # Reset the post_process to True
            self.post_process = True
        else:
            hidden_states = GPTModel.forward(self, *args, labels=None, **kwargs)

        return hidden_states

    def _medusa_forward(self, hidden_states):
        draft_logits = []
        # Medusa heads forward. We want to run through all the heads just to make sure all modules
        # are exercised during calibration.
        for i, head in enumerate(self.medusa_heads):
            new_logits, _ = head(hidden_states)

            draft_logits.append(new_logits)

        return draft_logits

    def forward(self, *args, labels: torch.Tensor = None, **kwargs):
        """Forward pass of the Medusa GPTModel.

        Returns:
            torch.Tensor: If labels are provided, then return lm_loss of all heads. Otherwise,
                return the original logits.
        """
        hidden_states = self._base_model_forward(*args, labels=labels, **kwargs)

        if not self.post_process:
            return hidden_states

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        # Original output logits
        logits, _ = self.output_layer(hidden_states, weight=output_weight)

        draft_logits = self._medusa_forward(hidden_states)

        if self.medusa_report_acc and labels is not None:
            acc = []
            for i, _ in enumerate(self.medusa_heads):
                gathered_logits = gather_from_tensor_model_parallel_region(draft_logits[i])
                medusa_top1 = gathered_logits.transpose(0, 1).argmax(dim=-1)[:, : -(1 + i)]
                medusa_labels = labels[:, 1 + i :]
                top1_p = torch.eq(medusa_labels, medusa_top1).sum() / medusa_top1.numel()
                acc.append(top1_p)

            if get_tensor_model_parallel_rank() == 0:
                print("Medusa Training Accuracy: {}".format(acc))

        # Return the original logits untouched.
        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        # Base model loss
        # If medusa_freeze_base_model is set to True,
        # the base model is frozen .
        loss = self.compute_language_model_loss(labels, logits)
        # Medusa loss
        for i, _ in enumerate(self.medusa_heads):
            medusa_labels = labels[:, 1 + i :]
            medusa_loss = self.compute_language_model_loss(
                medusa_labels, draft_logits[i][: -(1 + i), :]
            )
            loss[:, 1 + i :] += medusa_loss

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

    def pseudo_speculative_generate(self, *args, tp=1, steps=1, **kwargs):
        """Pseudo generate of the Medusa GPTModel.

        Returns:
            base_token (torch.Tensor): token from base model
            draft_tokens (torch.Tensor): draft tokens from medusa heads
        """
        hidden_states = self._base_model_forward(*args, labels=None, **kwargs)

        if not self.post_process:
            return hidden_states

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        # Original output logits
        logits, _ = self.output_layer(hidden_states, weight=output_weight)

        draft_logits = self._medusa_forward(hidden_states)

        logits = gather_from_tensor_model_parallel_region(logits.transpose(0, 1).contiguous())
        draft_logits = [
            gather_from_tensor_model_parallel_region(logit.transpose(0, 1).contiguous())
            for logit in draft_logits
        ]

        # [b, s]
        base_token = logits[:, -1:].argmax(dim=-1)
        draft_tokens = [logit[:, -1:].argmax(dim=-1) for logit in draft_logits]
        draft_tokens = torch.cat(draft_tokens, dim=-1)

        return base_token, draft_tokens


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
                # (PP, TP, DP)
                replica_id=(1, 0, get_data_parallel_rank(with_context_parallel=True)),
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

    def __init__(
        self,
        config,
        num_layers: int,
        transformer_layer_spec,
        use_last_layernorm,
        use_mtp_layernorm=False,
        bias=True,
    ):
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

        # TODO: remove this once mcore 0.12 is released
        from megatron.core.post_training.modelopt.layers import Linear

        self._use_mtp_layernorm = use_mtp_layernorm

        eagle_config = copy.deepcopy(config)
        eagle_config.num_layers = num_layers
        eagle_config.pipeline_model_parallel_size = 1
        # Unset the uneven PP config.
        eagle_config.num_layers_in_first_pipeline_stage = None
        eagle_config.num_layers_in_last_pipeline_stage = None

        # If heterogenous layers (e.g. DeepSeek), transformer_layer_spec is a
        # TransformerBlockSubmodules instead. We use the last layer_specs.
        if "TransformerBlockSubmodules" in str(type(transformer_layer_spec)):
            eagle_transformer_layer_spec = copy.deepcopy(transformer_layer_spec.layer_specs[-1])
        else:
            eagle_transformer_layer_spec = copy.deepcopy(transformer_layer_spec)

        # Force TransformerLayer in case RealQuantTransformerLayer was used.
        eagle_transformer_layer_spec.module = TransformerLayer

        if self._use_mtp_layernorm:
            self.enorm = TENorm(
                eagle_config, eagle_config.hidden_size, eagle_config.layernorm_epsilon
            )
            self.hnorm = TENorm(
                eagle_config, eagle_config.hidden_size, eagle_config.layernorm_epsilon
            )

        if config.use_cpu_initialization:
            device = "cpu"
        else:
            device = torch.cuda.current_device()

        # This linear was previously a ColumnParallelLinear. We changed it to a normal linear
        # since ColumnParallelLinear will have try to gather the input sequence when sequence
        # parallel is used and does not allow gathering the outputs.
        self.fc = Linear(
            eagle_config.hidden_size * 2,
            eagle_config.hidden_size,
            config=eagle_config,
            init_method=(lambda w: None),  # not used
            bias=bias,
        ).to(device)

        # The base model may have uneven PP.
        # We need to reset those configs.
        eagle_config.num_layers_in_first_pipeline_stage = None
        eagle_config.num_layers_in_last_pipeline_stage = None

        # Eagle does not use the final_layernorm in decoder. It distills the base model
        # final_layernorm to eagle_module hidden_states directly.
        self.decoder = EagleTransformerBlock(
            config=eagle_config,
            spec=eagle_transformer_layer_spec,
            post_layer_norm=use_last_layernorm,
            pre_process=True,
            post_process=True,
        ).to(device)

    def forward(
        self,
        embeddings: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
    ) -> torch.Tensor:
        """Forward function."""
        if self._use_mtp_layernorm:
            embeddings = self.enorm(embeddings)
            hidden_states = self.hnorm(hidden_states)

        decoder_input = torch.cat((embeddings, hidden_states), dim=-1)
        decoder_input = self.fc(decoder_input)[0]

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
        self._register_temp_attribute("eagle_self_logit_distillation", True)
        self._register_temp_attribute("eagle_freeze_base_model", True)
        self._register_temp_attribute("calibration_mode", False)

    def modify(
        self,
        eagle_num_layers=0,
        use_input_layernorm_in_first_layer=True,
        use_last_layernorm=False,
        use_mtp_layernorm=False,
        eagle_self_logit_distillation=True,
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
                use_mtp_layernorm=use_mtp_layernorm,
            )

        # Eagle loss functions
        self.eagle_loss = torch.nn.SmoothL1Loss(reduction="none")
        self.kld = logits_kld_loss

    def _base_model_forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
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

        if mcore_version_higher_than("0.9.0"):
            extra_kwargs = {"packed_seq_params": None}
        else:
            extra_kwargs = {}

        if self.config.multi_latent_attention:
            # For MLA, rotary_pos_emb is computed per attention.
            rotary_pos_emb = None
        else:
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

        return hidden_states

    def _eagle_forward(
        self,
        input_ids,
        hidden_states,
        position_ids,
        attention_mask,
        rotary_pos_emb,
        output_weight,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
    ):
        eagle_embeddings = self.embedding(
            input_ids=input_ids,
            position_ids=position_ids,
        )

        eagle_hidden_states = self.eagle_module(
            eagle_embeddings,
            hidden_states,
            attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )
        eagle_logits, _ = self.output_layer(eagle_hidden_states, weight=output_weight)

        return eagle_hidden_states, eagle_logits

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
        hidden_states = self._base_model_forward(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input,
            inference_params,
            packed_seq_params,
            extra_block_kwargs,
        )

        if not self.post_process:
            return hidden_states

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits_sbh, _ = self.output_layer(hidden_states, weight=output_weight)

        if inference_params is None or self.calibration_mode:
            eagle_ids = torch.cat(
                (
                    input_ids[:, 1:],
                    torch.zeros(
                        input_ids.shape[0], 1, dtype=input_ids.dtype, device=input_ids.device
                    ),
                ),
                dim=-1,
            )

            eagle_position_ids = (
                torch.cat(
                    (
                        position_ids[:, 1:],
                        torch.zeros(
                            position_ids.shape[0],
                            1,
                            dtype=position_ids.dtype,
                            device=position_ids.device,
                        ),
                    ),
                    dim=-1,
                ),
            )

            # pos_id        0 1 2 3 4 5 6 7
            # tokens        x x x x x x x x
            # hidden       [x x x x x x x -] ->
            # logits       [x x x x x x x -] -> torch.cat(hidden, emb(logits.argmax(dim=-1))) inputs to eagle_input
            # labels        x[x x x x x x x]
            #
            # eagle_input     x x x x x x x x
            # eagle_mask      1 1 1 1 1 1 1 0
            # eagle_hidden   [x x x x x x x]x
            # eagle_logits   [x x x x x x x]x
            eagle_attention_mask = attention_mask.clone().detach()
            eagle_attention_mask[:, :, :-1, :-1] = attention_mask[:, :, 1:, 1:]
            eagle_attention_mask[:, :, -1, :] = False
            eagle_attention_mask[:, :, :, -1] = False

            if self.config.multi_latent_attention:
                # For MLA, rotary_pos_emb is computed per attention.
                rotary_pos_emb = None
            else:
                rotary_pos_emb = self.rotary_pos_emb(eagle_ids.shape[-1])

            eagle_hidden_states, eagle_logits = self._eagle_forward(
                eagle_ids,
                hidden_states,
                eagle_position_ids,
                eagle_attention_mask,
                rotary_pos_emb,
                output_weight,
                inference_params,
                packed_seq_params,
                extra_block_kwargs,
            )

        # If labels are not provided, return the original logits. We only return after
        # all eagle weights have been exercised for quantization calibration purpose.
        if labels is None:
            return logits_sbh.transpose(0, 1).contiguous()

        # Base model loss
        # If eagle_freeze_base_model is set to True,
        # the base model is frozen .
        loss = self.compute_language_model_loss(labels, logits_sbh)

        if self.config.sequence_parallel:
            # [s, b, h] = gather([s/TP, b, h])
            gathered_hidden_states = gather_from_sequence_parallel_region(hidden_states)
            gathered_eagle_hidden_states = gather_from_sequence_parallel_region(eagle_hidden_states)
        else:
            gathered_hidden_states = hidden_states
            gathered_eagle_hidden_states = eagle_hidden_states

        # Compute hidden state regression loss
        regression_loss = (
            self.eagle_loss(
                gathered_eagle_hidden_states[:-1, :, :],
                gathered_hidden_states[1:, :, :],
            )
            .mean(dim=-1)
            .transpose(0, 1)
        )

        # Compute eagle lm loss (classification loss) or KLDivergence
        if self.eagle_self_logit_distillation:
            lm_loss = self.kld(eagle_logits[:-1, :, :], logits_sbh[1:, :, :])
        else:
            lm_loss = self.compute_language_model_loss(labels[:, 1:], eagle_logits[:-1, :, :])

        loss[:, 1:] += lm_loss + 10.0 * regression_loss

        acc = []
        if self.eagle_report_acc:
            with torch.no_grad():
                gathered_logits = gather_from_tensor_model_parallel_region(eagle_logits[:-1, :, :])
                eagle_top1 = gathered_logits.transpose(0, 1).argmax(dim=-1)
                top1_p = torch.eq(labels[:, 1:], eagle_top1).sum() / eagle_top1.numel()
                acc.append(top1_p)

            if get_tensor_model_parallel_rank() == 0:
                print("EAGLE Training Accuracy: {}".format(acc))

        return loss

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

            eagle_embeddings = self.embedding(
                input_ids=eagle_ids,
                position_ids=eagle_position_ids,
            )

            new_hidden_states = self.eagle_module(
                eagle_embeddings,
                eagle_hidden_states,
                eagle_attn_mask,
                rotary_pos_emb=eagle_rotary_pos_emb,
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

    def pseudo_speculative_generate(
        self,
        input_ids: torch.Tensor,
        tp: int,
        steps: int = 1,
    ):
        """Pseudo generate of the EAGLE GPTModel.

        Returns:
            base_token (torch.Tensor): token from base model
            draft_tokens (torch.Tensor): draft tokens from eagle module
        """
        padded_input_ids, seq_len = right_padding(input_ids, tp)

        attention_mask, position_ids = get_default_attention_mask_and_position_ids(padded_input_ids)

        hidden_states = self._base_model_forward(
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

        # Removing the padding
        logits_sbh = logits_sbh[:seq_len, :, :]
        hidden_states = hidden_states[:seq_len, :, :]

        base_token = (
            gather_from_tensor_model_parallel_region(logits_sbh)[-1:, :, :]
            .argmax(dim=-1)
            .transpose(0, 1)
        )

        # Early return
        if steps < 1:
            return base_token, None

        eagle_ids = torch.cat((input_ids[:, 1:], base_token), dim=-1)

        draft_tokens = []

        for _ in range(steps):
            padded_eagle_ids, seq_len, padded_hidden_states = right_padding(
                eagle_ids, tp, hidden_states
            )
            eagle_attention_mask, eagle_position_ids = get_default_attention_mask_and_position_ids(
                padded_eagle_ids
            )

            if self.config.multi_latent_attention:
                # For MLA, rotary_pos_emb is computed per attention.
                rotary_pos_emb = None
            else:
                rotary_pos_emb = self.rotary_pos_emb(padded_eagle_ids.shape[-1])

            eagle_hidden_states, eagle_logits = self._eagle_forward(
                padded_eagle_ids,
                padded_hidden_states,
                eagle_position_ids,
                eagle_attention_mask,
                rotary_pos_emb,
                output_weight,
            )

            eagle_logits = eagle_logits[:seq_len, :, :]
            eagle_hidden_states = eagle_hidden_states[:seq_len, :, :]

            draft_token = (
                gather_from_tensor_model_parallel_region(eagle_logits)[-1:, :, :]
                .argmax(dim=-1)
                .transpose(0, 1)
            )
            draft_tokens.append(draft_token)

            eagle_ids = torch.cat((eagle_ids, draft_token), dim=-1)
            hidden_states = torch.cat((hidden_states, eagle_hidden_states[-1:, :, :]), dim=0)

        draft_tokens = torch.cat(draft_tokens, dim=-1)

        return base_token, draft_tokens


@MTPDMRegistry.register({GPTModel: "megatron.core.models.gpt.GPTModel"})
class _DynamicMTPGPTModel(MTPModel):
    """A ``megatron.core.models.gpt.GPTModel`` model with dynamic hyperparams."""

    def _setup(self):
        super()._setup()
        self._register_temp_attribute("mtp_self_logit_distillation", True)
        self._register_temp_attribute("mtp_freeze_base_model", True)
        self._register_temp_attribute("calibration_mode", False)

    def modify(
        self,
        mtp_num_layers=0,
        mtp_num_module=0,
        mtp_freeze_list=[],
        use_last_layernorm=False,
        mtp_self_logit_distillation=True,
        mtp_freeze_base_model=True,
        mtp_report_acc=True,
    ):
        if self.config.pipeline_model_parallel_size > 1:
            warnings.warn(
                "Pipeline parallelism detected! _DynamicMTPGPTModel only supports "
                "pipeline parallelism during TensorRT-LLM checkpoint export."
            )
        super().modify(
            mtp_num_layers=mtp_num_layers,
            mtp_num_module=mtp_num_module,
            mtp_freeze_list=mtp_freeze_list,
            use_last_layernorm=use_last_layernorm,
        )
        self.mtp_report_acc = mtp_report_acc
        self.mtp_self_logit_distillation = mtp_self_logit_distillation
        self.mtp_freeze_base_model = mtp_freeze_base_model

        if self.position_embedding_type != "rope":
            raise ValueError("For MTP, only rotary embedding is supported")

        if not self.pre_process and self.post_process:
            self.embedding = EagleLanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=self.position_embedding_type,
            )

        # Freeze all parameters
        if self.mtp_freeze_base_model:
            for name, param in self.named_parameters():
                param.requires_grad = False

        # Only the last PP stage has the additional projection and decoder layer.
        # This is to simplify the export.
        if self.post_process:
            self.mtp = torch.nn.ModuleList()
            for i in range(self.mtp_num_module):
                mtp = EagleModule(
                    self.config,
                    self.mtp_num_layers,
                    self.transformer_layer_spec,
                    self.use_last_layernorm,
                    use_mtp_layernorm=True,
                    bias=False,
                )
                if i in self.mtp_freeze_list:
                    for name, param in mtp.named_parameters():
                        param.requires_grad = False
                self.mtp.append(mtp)

        self.kld = logits_kld_loss

    def _base_model_forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
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

        if mcore_version_higher_than("0.9.0"):
            extra_kwargs = {"packed_seq_params": None}
        else:
            extra_kwargs = {}

        if self.config.multi_latent_attention:
            # For MLA, rotary_pos_emb is computed per attention.
            rotary_pos_emb = None
        else:
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

        return hidden_states

    def _mtp_forward(
        self,
        index,
        input_ids,
        hidden_states,
        position_ids,
        attention_mask,
        rotary_pos_emb,
        output_weight,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
    ):
        mtp_embeddings = self.embedding(
            input_ids=input_ids,
            position_ids=position_ids,
        )
        hidden_states = self.mtp[index](
            mtp_embeddings,
            hidden_states,
            attention_mask,  # TODO (chenhany): this may needs some fix
            rotary_pos_emb=rotary_pos_emb,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )
        mtp_logits, _ = self.output_layer(hidden_states, weight=output_weight)

        return hidden_states, mtp_logits

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
        hidden_states = self._base_model_forward(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input,
            inference_params,
            packed_seq_params,
            extra_block_kwargs,
        )

        if not self.post_process:
            return hidden_states

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits_sbh, _ = self.output_layer(hidden_states, weight=output_weight)

        if inference_params is None or self.calibration_mode:
            draft_logits = []
            for i in range(self.mtp_num_module):
                mtp_ids = torch.cat(
                    (
                        input_ids[:, 1 + i :],
                        torch.zeros(
                            input_ids.shape[0],
                            1 + i,
                            dtype=input_ids.dtype,
                            device=input_ids.device,
                        ),
                    ),
                    dim=-1,
                )
                padding_zeros = torch.zeros(
                    position_ids.shape[0],
                    1 + i,
                    dtype=position_ids.dtype,
                    device=position_ids.device,
                )
                mtp_position_ids = torch.cat((position_ids[:, 1 + i :], padding_zeros), dim=1)

                mtp_attention_mask = attention_mask.clone().detach()
                mtp_attention_mask[:, :, : -(1 + i), : -(1 + i)] = attention_mask[
                    :, :, 1 + i :, 1 + i :
                ]
                mtp_attention_mask[:, :, -(1 + i) :, :] = False
                mtp_attention_mask[:, :, :, -(1 + i) :] = False

                if self.config.multi_latent_attention:
                    # For MLA, rotary_pos_emb is computed per attention.
                    rotary_pos_emb = None
                else:
                    rotary_pos_emb = self.rotary_pos_emb(mtp_ids.shape[-1])

                hidden_states, mtp_logits = self._mtp_forward(
                    i,
                    mtp_ids,
                    hidden_states,
                    mtp_position_ids,
                    mtp_attention_mask,
                    rotary_pos_emb,
                    output_weight,
                    inference_params,
                    packed_seq_params,
                    extra_block_kwargs,
                )
                draft_logits.append(mtp_logits)

        # If labels are not provided, return the original logits. We only return after
        # all mtp weights have been exercised for quantization calibration purpose.
        if labels is None:
            return logits_sbh.transpose(0, 1).contiguous()

        # Base model loss
        # If mtp_freeze_base_model is set to True,
        # the base model is frozen .
        loss = self.compute_language_model_loss(labels, logits_sbh)

        for i, mtp_logits in enumerate(draft_logits):
            # Compute lm loss (classification loss) or KLDivergence
            if self.mtp_self_logit_distillation:
                mtp_loss = self.kld(
                    mtp_logits[: -(1 + i), :, :],
                    logits_sbh[1 + i :, :, :],
                )
            else:
                mtp_loss = self.compute_language_model_loss(
                    labels[:, 1 + i :], mtp_logits[: -(1 + i), :, :]
                )

            loss[:, 1 + i] += mtp_loss

            acc = []
            if self.mtp_report_acc:
                with torch.no_grad():
                    gathered_logits = gather_from_tensor_model_parallel_region(mtp_logits)
                    mtp_top1 = gathered_logits.transpose(0, 1).argmax(dim=-1)
                    mtp_top1 = mtp_top1[:, : -(1 + i)]
                    top1_p = torch.eq(labels[:, 1 + i :], mtp_top1).sum() / mtp_top1.numel()
                    acc.append(top1_p)

                if get_tensor_model_parallel_rank() == 0:
                    print("MTP_{} Training Accuracy: {}".format(i, acc))

        return loss

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict = None
    ) -> ShardedStateDict:
        """Override the shared_state_dict to take care mtp."""
        assert not sharded_offsets, "Unexpected sharded offsets"

        sharded_state_dict = GPTModel.sharded_state_dict(self, prefix, sharded_offsets, metadata)

        if not hasattr(self, "mtp") or self.mtp is None:
            return sharded_state_dict

        # This is a remedy for nn.ModuleList. GPTModel.sharded_state_dict() is calling into
        # MegatronModule.sharded_state_dict() which requires all children to implement
        # sharded_state_dict(). mtp is an nn.ModuleList which only has state_dict()
        # implemented. As a result, all the submodules will not be sharded.
        #
        # The remedy is to pop all mtp* out and call the EagleModule sharded_state_dict()
        # again to populate the correct sharded_staet_dict.
        extra_keys = []
        for key, val in sharded_state_dict.items():
            if "mtp" in key:
                extra_keys += [key]
        for key in extra_keys:
            sharded_state_dict.pop(key, None)

        layer_prefix = f"{prefix}mtp."
        for i, layer in enumerate(self.mtp):
            layer_sharded_state_dict = layer.sharded_state_dict(f"{layer_prefix}{i}.", [], metadata)
            sharded_state_dict.update(layer_sharded_state_dict)
        return sharded_state_dict

    def pseudo_speculative_generate(
        self,
        input_ids: torch.Tensor,
        tp: int,
        steps: int = 1,
    ):
        """Pseudo generate of the MTP GPTModel.

        Returns:
            base_token (torch.Tensor): token from base model
            draft_tokens (torch.Tensor): draft tokens from MTP
        """
        padded_input_ids, seq_len = right_padding(input_ids, tp)

        attention_mask, position_ids = get_default_attention_mask_and_position_ids(padded_input_ids)

        hidden_states, logits_sbh, output_weight = self._base_model_forward(
            padded_input_ids,
            position_ids,
            attention_mask,
        )

        # Removing the padding
        logits_sbh = logits_sbh[:seq_len, :, :]
        hidden_states = hidden_states[:seq_len, :, :]

        base_token = (
            gather_from_tensor_model_parallel_region(logits_sbh)[-1:, :, :]
            .argmax(dim=-1)
            .transpose(0, 1)
        )
        mtp_ids = torch.cat((input_ids[:, 1:], base_token), dim=-1)

        draft_tokens = []
        for i in range(self.mtp_num_module):
            padded_mtp_ids, seq_len, padded_hidden_states = right_padding(
                mtp_ids, tp, hidden_states
            )
            mtp_attention_mask, mtp_position_ids = get_default_attention_mask_and_position_ids(
                padded_mtp_ids
            )
            if self.config.multi_latent_attention:
                # For MLA, rotary_pos_emb is computed per attention.
                rotary_pos_emb = None
            else:
                rotary_pos_emb = self.rotary_pos_emb(padded_mtp_ids.shape[-1])

            mtp_hidden_states, mtp_logits = self._mtp_forward(
                i,
                padded_mtp_ids,
                padded_hidden_states,
                mtp_position_ids,
                mtp_attention_mask,
                rotary_pos_emb,
                output_weight,
            )

            mtp_logits = mtp_logits[:seq_len, :, :]
            mtp_hidden_states = mtp_hidden_states[:seq_len, :, :]

            draft_token = (
                gather_from_tensor_model_parallel_region(mtp_logits)[-1:, :, :]
                .argmax(dim=-1)
                .transpose(0, 1)
            )
            draft_tokens.append(draft_token)

            mtp_ids = torch.cat((mtp_ids, draft_token), dim=-1)
            hidden_states = torch.cat((hidden_states, mtp_hidden_states[-1:, :, :]), dim=0)

        draft_tokens = torch.cat(draft_tokens, dim=-1)

        return base_token, draft_tokens


class MegatronARValidation(AcceptanceRateValidation):
    """This is the subclass for megatron model AR validation."""

    def get_ground_truth(self, input_ids, osl):
        """This function returns ground truth output tokens from the base model."""
        input_ids = copy.deepcopy(input_ids)
        for _ in range(osl):
            input_id, _ = self.model.pseudo_speculative_generate(input_ids, tp=self.tp, steps=0)
            input_ids = torch.cat((input_ids, input_id), dim=-1)
            if input_id[0, 0] == self.end_token:
                break
        return input_ids
