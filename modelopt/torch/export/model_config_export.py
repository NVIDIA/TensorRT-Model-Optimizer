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

"""Code that export optimized models to the TensorRT-LLM checkpoint."""

import copy
import json
import math
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Optional, Union
from warnings import warn

import torch
import torch.nn as nn
from safetensors.torch import save_file

from modelopt.torch.utils import distributed as dist
from modelopt.torch.utils import import_plugin

from .layer_utils import (
    build_conv_config,
    build_decoder_config,
    build_embedding_config,
    build_layernorm_config,
    build_linear_config,
    build_medusa_heads_config,
    check_model_compatibility,
    get_dtype,
    get_transformer_layers,
    is_conv,
    is_decoder_list,
    is_embedding,
    is_layernorm,
    is_linear,
)
from .model_config import QUANTIZATION_INT4_AWQ, QUANTIZATION_W4A8_AWQ, ModelConfig
from .model_config_utils import (
    merge_qkv,
    model_config_to_dict,
    naive_quantization,
    pack_linear_weights,
    split_config_and_weights,
)
from .postprocess import (
    check_weight_shape_valid,
    pad_embedding_lm_head,
    postprocess_model_config,
    postprocess_tensors,
    update_lm_head_quantization,
)
from .quant_utils import get_quantization_format, process_layer_quant_config
from .tensorrt_llm_utils import (
    convert_to_tensorrt_llm_config,
    is_tensorrt_llm_0_8_or_9,
    prepare_enc_dec_decoder_layer,
    prepare_enc_dec_export_dir,
)

has_mcore = False
with import_plugin("megatron", verbose=False):
    from megatron.core.models.gpt import GPTModel as MCoreGPTModel
    from megatron.core.transformer.module import MegatronModule

    has_mcore = True

__all__ = ["export_tensorrt_llm_checkpoint", "torch_to_tensorrt_llm_checkpoint"]


def torch_to_tensorrt_llm_checkpoint(
    model: nn.Module,
    decoder_type: str,
    dtype: Optional[torch.dtype] = None,
    inference_tensor_parallel: int = 0,
    inference_pipeline_parallel: int = 1,
    naive_fp8_quantization: bool = False,
    workspace_path: Optional[Union[Path, str]] = None,
) -> Iterator[tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, Any]]]:
    """Converts the torch model to the TensorRT-LLM checkpoint per GPU rank.

    TensorRT-LLM checkpoint is the LLM model format that can be used by the TensorRT-LLM build API.
    for the engine building process.
    https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/architecture/checkpoint.md

    Args:
        model: the torch model.
        decoder_type: the type of the decoder, e.g. gpt, gptj, llama.
        dtype: the weights data type to export the unquantized layers or the default model data type if None.
        inference_tensor_parallel: The target inference time tensor parallel.
            We will merge or split the calibration tensor parallelism to inference.
            Default is 0, meaning using the calibration without manual config merge or split.
        inference_pipeline_parallel: The target inference time pipeline parallel.
            We will merge or split the calibration pipeline parallelism to inference.
            Default is 1, meaning no pipeline parallelism.
        naive_fp8_quantization: Quantize the model naively to FP8 without calibration.
            All scaling factors are set to 1.
        workspace_path: the path to the NFS directory for postprocess cross rank communication.

    Yields:
        A tuple of
            tensorrt_llm_config: A dict that maps to the ``PretrainedConfig`` in TensorRT-LLM.
            https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/models/modeling_utils.py
            weights: A dict that stores all model weights and scaling factors for each rank.
            per_layer_quantization: A dict that contains layer-wise quantization information for all quantized layers
            for mixed_precision, empty dictionary otherwise.
    """
    if dtype is None:
        dtype = get_dtype(model)

    if dtype not in [torch.float16, torch.bfloat16]:
        warn(
            f"dtype {dtype} not fully compatible with TensorRT-LLM optimizations, Default to float16."
        )
        dtype = torch.float16

    architecture = ""
    hf_config = None
    if has_mcore and isinstance(model, MegatronModule):
        if not isinstance(model, MCoreGPTModel):
            raise ValueError("Only megatron.core.models.gpt.GPTModel is supported!")
        # MCoreGPTModel.config has type TransformerConfig
        #
        # We choose to deepcopy here since TransformerConfig deserialization is sensitive to
        # additional attributes.
        model_metadata_config = copy.deepcopy(model.config.__dict__)
        vocab_size = model.vocab_size
        model_metadata_config["max_position_embeddings"] = model.max_position_embeddings
        model_metadata_config["rotary_percent"] = model.rotary_percent
        model_metadata_config["rotary_base"] = getattr(model, "rotary_base", 10000)
        rope_scaling = getattr(model, "rotary_scaling", False)
        if rope_scaling:
            model_metadata_config["rope_scaling"] = {"type": "llama3"}
    elif hasattr(model, "config"):
        # Huggingface models
        model_metadata_config = model.config.__dict__
        vocab_size = model.config.vocab_size
        hf_config = model.config
        architecture = model.config.architectures[0]

        # For Baichuan 13B, we check if alibi is used with the alibi_mask property.
        if hasattr(model, "model") and hasattr(model.model, "alibi_mask"):
            model_metadata_config["alibi"] = True

        # For MPT, DBRX
        for config_key in ["attn_config", "ffn_config"]:
            config_value = model_metadata_config.get(config_key, None)
            if config_value:
                model_metadata_config.update(
                    config_value if isinstance(config_value, dict) else config_value.to_dict()
                )

    elif hasattr(model, "cfg"):
        # NeMo MegatronGPTModel
        model_metadata_config = dict(model.cfg)
        vocab_size = model.tokenizer.vocab_size
    else:
        raise ValueError("Cannot find valid model metadata config in model")

    if "multi_query_group_num" in model_metadata_config.keys():
        if model_metadata_config["multi_query_group_num"] % inference_tensor_parallel != 0:
            raise ValueError(
                "Cannot divide {} kv_heads into {} gpus".format(
                    model_metadata_config["multi_query_group_num"],
                    inference_tensor_parallel,
                )
            )

    training_pipeline_parallel = model_metadata_config.get("pipeline_model_parallel_size", 1)
    training_tensor_parallel = dist.size() // training_pipeline_parallel
    model_metadata_config["training_pipeline_parallel"] = training_pipeline_parallel
    model_metadata_config["training_tensor_parallel"] = training_tensor_parallel

    if "make_vocab_size_divisible_by" in model_metadata_config:
        # For some nemo models, the vocab_size is pre-padded.
        # We calculate the pre-padded vocab_size with this config: make_vocab_size_divisible_by.
        make_vocab_size_divisible_by = model_metadata_config["make_vocab_size_divisible_by"]
        make_vocab_size_divisible_by_with_tp = (
            make_vocab_size_divisible_by * training_tensor_parallel
        )
        vocab_size = int(
            math.ceil(vocab_size / make_vocab_size_divisible_by_with_tp)
            * make_vocab_size_divisible_by_with_tp
        )
        print(
            f"the new vocab_size is updated: {vocab_size}, make_vocab_size_divisible_by"
            f" {make_vocab_size_divisible_by}, training_tensor_parallel"
            f" {training_tensor_parallel}."
        )

    models = [model]
    if decoder_type in ["t5"]:
        model_lm_head = model.lm_head
        # For T5 model with encoder and decoder, we process the checkpoint separately.
        models = [model.encoder, model.decoder]

    elif decoder_type in ["whisper"]:
        model_lm_head = model.proj_out
        models = [model.model.encoder, model.model.decoder]

    for model in models:
        transformer_layers = get_transformer_layers(model)
        if training_pipeline_parallel == 1:
            compatible, has_position_embedding, has_embedding_layernorm = check_model_compatibility(
                transformer_layers
            )
        else:
            # For Megatron models with more than one PP,
            # we skip the compatibility check as not all ranks have the full model.
            # For Megatron Core GPTModel, both nemotron and llama do not have position embedding
            # nor embedding layernorm.
            compatible = len(transformer_layers) > 0
            has_position_embedding = False
            has_embedding_layernorm = False
        assert compatible, "The model is not supported"

        config = ModelConfig(
            architecture=architecture,
            dtype=str(dtype).split(".")[1],
            rank=dist.rank(),
            tensor_parallel=training_tensor_parallel,
            pipeline_parallel=training_pipeline_parallel,
            vocab_size=vocab_size,
        )

        # For Encoder-Decoder Model like T5
        if decoder_type in ["t5"]:
            if model.is_decoder is False:
                config.enc_dec = "enc"
                model_metadata_config["enc_dec"] = "enc"
            else:
                config.enc_dec = "dec"
                model_metadata_config["enc_dec"] = "dec"
        elif decoder_type in ["whisper"]:
            if type(model).__name__ == "WhisperEncoder":
                model_metadata_config["enc_dec"] = config.enc_dec = "enc"
                has_position_embedding = True
            else:
                model_metadata_config["enc_dec"] = config.enc_dec = "dec"

        # GLM has a different handling of word_embeddings.
        if decoder_type == "glm":
            model = model.glm
            transformer_layers.insert(0, model.word_embeddings)

        # Build the full model_config dict layer by layer.
        for module in transformer_layers:
            if is_embedding(module):
                if config.vocab_embedding is None and not (
                    decoder_type == "whisper" and model_metadata_config["enc_dec"] == "enc"
                ):
                    # We assume the first embedding in the list the vocab_embedding.

                    normalization_constant = 1
                    # Normalize vocab embedding for gemma.
                    if (
                        decoder_type == "gemma" and is_tensorrt_llm_0_8_or_9()
                    ) or decoder_type == "recurrentgemma":
                        normalization_constant = model_metadata_config["hidden_size"] ** 0.5

                    config.vocab_embedding = build_embedding_config(
                        module, normalization_constant=normalization_constant
                    )
                elif has_position_embedding and config.position_embedding is None:
                    config.position_embedding = build_embedding_config(module)
                elif decoder_type == "glm":
                    config.block_embedding = build_embedding_config(module)
            elif is_decoder_list(module):
                layers = []
                for idx, layer in enumerate(module.children()):
                    # Special process due to T5 model structure's specialty
                    if decoder_type in ["t5"]:
                        layer = layer.layer
                    layer_config = build_decoder_config(
                        layer,
                        model_metadata_config,
                        decoder_type,
                        tp_size=inference_tensor_parallel,
                    )
                    if decoder_type in ["deci"]:
                        block_config = model_metadata_config["block_configs"][idx]
                        layer_config.block_config = asdict(block_config)
                    # Special process for each decoder layer of Encoder-Decoder Model
                    if decoder_type in ["t5"]:
                        prepare_enc_dec_decoder_layer(
                            layer_config,
                            model.config,
                            model_metadata_config["enc_dec"],
                            layers,
                        )
                    layers.append(layer_config)
                config.layers = layers
            elif is_layernorm(module):
                if has_embedding_layernorm and config.ln_embed is None:
                    # Assume embedding_layernorm is placed before the ln_f.
                    config.ln_embed = build_layernorm_config(module)
                else:
                    config.ln_f = build_layernorm_config(module)
            elif is_linear(module):
                if model_metadata_config.get("share_embeddings_and_output_weights", False):
                    # NeMo/MCore models with shared embeddings - for example Gemma -
                    # the model head weight is None so we just skip processing
                    config.share_embedding_table = True
                    continue
                # TRT LLM forces the embedding table to be shared for the following models.
                force_share_embedding_table = decoder_type in ["gemma", "gemma2"]
                if force_share_embedding_table and torch.equal(
                    module.weight, config.vocab_embedding.weight
                ):
                    config.share_embedding_table = True
                else:
                    # This will update lm_head quantization config according to constraints from TRT-LLM
                    update_lm_head_quantization(config, module, inference_pipeline_parallel)
                    config.lm_head = build_linear_config(module, "column")
            elif is_conv(module):
                config.feature_extractor.append(build_conv_config(module))

        # For decoder of Encoder-Decoder model, it needs some encoder information
        if decoder_type in ["t5"]:
            if model_metadata_config["enc_dec"] == "dec":
                config.encoder_hidden_size = models[0].config.d_model
                config.encoder_head_size = models[0].config.d_kv
                config.encoder_num_heads = models[0].config.num_heads
        elif decoder_type in ["whisper"]:
            if model_metadata_config["enc_dec"] == "dec":
                config.encoder_hidden_size = models[0].config.d_model
                config.encoder_num_heads = models[0].config.encoder_attention_heads
                config.encoder_head_size = config.encoder_hidden_size // config.encoder_num_heads

        # For the training time PP, not all ranks will have the lm_head layer.
        if config.lm_head is None:
            if decoder_type in ["t5", "whisper"]:
                if decoder_type == "t5":
                    config.share_embedding_table = False
                if model_metadata_config["enc_dec"] == "dec":
                    config.lm_head = build_linear_config(model_lm_head, "column")
            elif training_pipeline_parallel == 1:
                # Models that share weights for lm_head and vocab_embedding
                assert decoder_type in [
                    "mpt",
                    "gpt2",
                    "gemma",
                    "gemma2",
                    "glm",
                ], f"lm_head not available for decoder {decoder_type}"
                config.share_embedding_table = True

        # Handle Medusa Heads
        # TODO (chenhany): post-processing is not implemented yet
        config.medusa_heads = build_medusa_heads_config(model)

        config.quantization = get_quantization_format(model)

        if config.quantization in [QUANTIZATION_INT4_AWQ, QUANTIZATION_W4A8_AWQ]:
            if config.vocab_size % 64 != 0:
                # TODO: Check if this works for Mixtral
                assert (
                    training_tensor_parallel == 1
                ), "We do not support padding for training time TP"
                print("Padding vocab_embedding and lm_head for AWQ weights export")
                pad_embedding_lm_head(config)
                if hf_config is not None:
                    hf_config.vocab_size = config.vocab_size

        check_weight_shape_valid(
            config,
            inference_tensor_parallel,
            training_tensor_parallel,
        )

        # If inference_tensor_parallel or inference_pipeline_parallel is different from world_size,
        # we try to merge or split the model configs based on the rank selected.
        if (
            inference_tensor_parallel > 0
            or inference_pipeline_parallel > 0
            or training_pipeline_parallel > 1
        ):
            model_configs = postprocess_model_config(
                config,
                inference_tensor_parallel,
                inference_pipeline_parallel,
                training_pipeline_parallel=training_pipeline_parallel,
                workspace_path=workspace_path,
            )
        else:
            model_configs = [config]

        for model_config in model_configs:
            assert model_config.rank >= 0, "Invalid model_config, postprocess_model_config fails."

            if not model_config.quantization and naive_fp8_quantization:
                naive_quantization(model_config)

            merge_qkv(model_config)
            pack_linear_weights(model_config)

            weights = {}
            # Layer config dict holds quantization format of each layer.
            # It also holds awq_block_size information for applicable layers.
            layer_config_dict = {}
            model_config_dict = model_config_to_dict(model_config)
            # We split the weights from model_config and save them separately as two files.
            split_config_and_weights(model_config_dict, weights, "transformer", layer_config_dict)
            # Process per layer quantization config dict
            per_layer_quantization = process_layer_quant_config(layer_config_dict)
            # We only export the json once across ranks as all jsons should be the same except for the rank.
            tensorrt_llm_config = convert_to_tensorrt_llm_config(
                model_config, weights.keys(), hf_config=hf_config
            )

            # Postprocess the tensors in the model_config.
            # Exporting the safetensors also allows the tensor to be a view.
            postprocess_tensors(
                weights,
                dtype=dtype,
                force_cpu=True,
                force_contiguous=True,
                force_non_view=False,
            )

            yield tensorrt_llm_config, weights, per_layer_quantization


def export_tensorrt_llm_checkpoint(
    model: nn.Module,
    decoder_type: str,
    dtype: Optional[torch.dtype] = None,
    export_dir: Union[Path, str] = tempfile.gettempdir(),
    inference_tensor_parallel: int = 0,
    inference_pipeline_parallel: int = 1,
    naive_fp8_quantization: bool = False,
    use_nfs_workspace: bool = False,
):
    """Exports the torch model to the TensorRT-LLM checkpoint and save to the export_dir.

    Args:
        model: the torch model.
        decoder_type: the type of the decoder, e.g. gpt, gptj, llama.
        dtype: the weights data type to export the unquantized layers or the default model data type if None.
        export_dir: the target export path.
        inference_tensor_parallel: The target inference time tensor parallel.
            We will merge or split the calibration tensor parallelism to inference.
            Default is 0, meaning using the calibration without manual config merge or split.
        inference_pipeline_parallel: The target inference time pipeline parallel.
            We will merge or split the calibration pipeline parallelism to inference.
            Default is 1, meaning no pipeline parallelism.
        inference_pipeline_parallel: The target inference time pipeline parallel.
        naive_fp8_quantization: Quantize the model naively to FP8 without calibration.
            All scaling factors are set to 1.
        use_nfs_workspace: if True, the an NFS workspace will be created under the export_dir and
            used as a shared memory for cross process/node communication.

    For tensorrt_llm deployment, save the representation under ``export_dir``.
    We will save the model_config as two files:

        * ``.json``: The nested dict that maps to the ``PretrainedConfig`` in TensorRT-LLM.
            https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/models/modeling_utils.py.
        * ``.safetensors``: The file for the list of weights as safetensors. Unique for each rank.
    """
    export_dir = Path(export_dir)
    export_root = export_dir
    export_dir.mkdir(parents=True, exist_ok=True)
    # Create a NFS workspace under the export folder which is also assumed to be NFS.
    workspace_path = None
    if use_nfs_workspace:
        workspace_path = export_dir.joinpath("workspace")
        workspace_path.mkdir(parents=True, exist_ok=True)

    try:
        for (
            tensorrt_llm_config,
            weights,
            per_layer_quantization,
        ) in torch_to_tensorrt_llm_checkpoint(
            model=model,
            decoder_type=decoder_type,
            dtype=dtype,
            inference_tensor_parallel=inference_tensor_parallel,
            inference_pipeline_parallel=inference_pipeline_parallel,
            naive_fp8_quantization=naive_fp8_quantization,
            workspace_path=workspace_path,
        ):
            exclude_modules = set()
            rank = tensorrt_llm_config["rank"]
            world_size = tensorrt_llm_config["mapping"]["world_size"]
            if tensorrt_llm_config["quantization"]:
                excluded_modules_rank = tensorrt_llm_config["quantization"].get(
                    "exclude_modules", {}
                )
                if excluded_modules_rank:
                    exclude_modules.update(excluded_modules_rank)
            # For T5 model
            if decoder_type in ["t5", "whisper"]:
                export_dir = prepare_enc_dec_export_dir(tensorrt_llm_config, export_root)
                export_dir = Path(export_dir)
                export_dir.mkdir(parents=True, exist_ok=True)
            if decoder_type in ["whisper"] and tensorrt_llm_config["quantization"].get(
                "exclude_modules"
            ):
                new_exclude_modules = []
                for module in tensorrt_llm_config["quantization"]["exclude_modules"]:
                    if tensorrt_llm_config["architecture"] == "WhisperEncoder":
                        module = module.replace("transformer.layers", "encoder_layers")
                        module = module.replace(
                            "transformer.position_embedding", "position_embedding"
                        )
                        module = module.replace("transformer.ln_f", "ln_post")
                        module = module.replace("transformer.feature_extractor.0", "conv1")
                        module = module.replace("transformer.feature_extractor.1", "conv2")
                        new_exclude_modules.append(module)
                    else:
                        module = module.replace("transformer.layers", "decoder_layers")
                        module = module.replace(
                            "transformer.position_embedding", "embedding.position_embedding"
                        )
                        module = module.replace(
                            "transformer.vocab_embedding", "embedding.vocab_embedding"
                        )
                        module = module.replace("transformer.ln_f", "final_layernorm")
                        new_exclude_modules.append(module)
                tensorrt_llm_config["quantization"]["exclude_modules"] = new_exclude_modules
                exclude_modules = set(new_exclude_modules)
            if rank == world_size - 1:
                # We only export the json once across ranks as all jsons should be the same except for the rank.
                # If auto_quant is used, save per layer quantization information in quant_cfg.json
                if per_layer_quantization:
                    # Update auto quant related information for quantization.json export
                    per_layer_quantization["kv_cache_quant_algo"] = tensorrt_llm_config[
                        "quantization"
                    ]["kv_cache_quant_algo"]

                    # Update auto quant related information for config.json export
                    # We remove group_size, has_zero_point, exclude_modules and pre_quant_scale information from config
                    tensorrt_llm_config["quantization"] = {
                        k: per_layer_quantization[k] for k in ("quant_algo", "kv_cache_quant_algo")
                    }

                    with open(export_dir / "quant_cfg.json", "w") as f:
                        json.dump(per_layer_quantization, f, indent=4)
                else:
                    # Excluded modules information is only included in non auto_quant case
                    tensorrt_llm_config["quantization"]["exclude_modules"] = list(exclude_modules)

                with open(export_dir / "config.json", "w") as f:
                    json.dump(tensorrt_llm_config, f, indent=4)

            # Hacky implementation for T5 for now
            new_weights = {}
            if decoder_type == "t5":
                for key in weights.keys():
                    if key == "transformer.vocab_embedding.weight":
                        new_key = "embedding.vocab_embedding.weight"
                    elif key.startswith("transformer.layers"):
                        # For encoder
                        if tensorrt_llm_config["architecture"] == "EncoderModel":
                            new_key = key.replace("transformer.layers", "encoder_layers")
                        # For decoder
                        else:
                            new_key = key.replace("transformer.layers", "decoder_layers")
                    elif key == "transformer.ln_f.weight":
                        new_key = "final_layernorm.weight"
                    elif key == "lm_head.weight":
                        new_key = key
                    if key.endswith("rel_attn_table.weight"):
                        new_key = new_key.replace("rel_attn_table.weight", "rel_attn_table")
                        if key.find(".0.") > 0:
                            # stores additional the relative attention for pre-compute feature in TRTLLM
                            new_weights["rel_attn_table"] = weights[key].clone()
                    new_weights[new_key] = weights[key]
                weights = new_weights
            # End of hacky implementation for T5 for now
            elif decoder_type == "whisper":
                for key, value in weights.items():
                    if tensorrt_llm_config["architecture"] == "WhisperEncoder":
                        new_key = key.replace("transformer.layers", "encoder_layers")
                        new_key = new_key.replace(
                            "transformer.position_embedding", "position_embedding"
                        )
                        new_key = new_key.replace("transformer.ln_f", "ln_post")
                        new_key = new_key.replace("transformer.feature_extractor.0", "conv1")
                        new_key = new_key.replace("transformer.feature_extractor.1", "conv2")
                    else:
                        new_key = key.replace("transformer.layers", "decoder_layers")
                        new_key = new_key.replace(
                            "transformer.position_embedding", "embedding.position_embedding"
                        )
                        new_key = new_key.replace(
                            "transformer.vocab_embedding", "embedding.vocab_embedding"
                        )
                        new_key = new_key.replace("transformer.ln_f", "final_layernorm")
                    new_weights[new_key] = value
                weights = new_weights
            weights_path = export_dir / f"rank{rank}.safetensors"
            save_file(weights, weights_path)

    except Exception as e:
        fallback_model_path = export_dir / f"modelopt_model.{dist.rank()}.pth"
        torch.save(model.state_dict(), fallback_model_path)
        warn(
            "Cannot export model to the model_config. The modelopt-optimized model state_dict"
            f" (including the quantization factors) is saved to {fallback_model_path} using"
            " torch.save for further inspection."
        )
        raise e
