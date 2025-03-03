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

"""Utils for TensorRT-LLM checkpoint export.

Some of the logics in this file are empirical and needs constant update if exceptions occur.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, Iterable, Optional
from warnings import warn

from .layer_utils import model_type_is_enc_dec
from .tensorrt_llm_type import LayerNormPositionType, LayerNormType, MLPType

if TYPE_CHECKING:
    from transformers import T5Config

from modelopt import __version__

from .model_config import (
    LAYERNORM_DEFAULT,
    LAYERNORM_RMS,
    QUANTIZATION_NONE,
    DecoderLayerConfig,
    MLPConfig,
    ModelConfig,
    MOEConfig,
)

logger = logging.getLogger(__name__)

# For NEMO and ENC/DEC models where the TensorRT-LLM model architecture and HF config are not aligned.
MODEL_NAME_TO_HF_ARCH_MAP = {
    "llama": "LlamaForCausalLM",
    "gemma": "GemmaForCausalLM",
    "gpt": "GPTForCausalLM",
    "enc": "EncoderModel",
    "dec": "DecoderModel",
    "mllama": "MLLaMAModel",
}


def is_tensorrt_llm_0_8_or_9():
    """Returns true if tensorrt_llm version is 0.8 or 0.9."""
    try:
        import tensorrt_llm

        return tensorrt_llm.__version__.startswith(("0.8", "0.9"))
    except Exception:
        return False


def _find_layernorm_type(model_config: ModelConfig):
    if model_config.ln_f:
        return model_config.ln_f.layernorm_type
    for layer in model_config.layers:
        if layer.input_layernorm:
            return layer.input_layernorm.layernorm_type
        if layer.post_layernorm:
            return layer.post_layernorm.layernorm_type
    return LAYERNORM_DEFAULT


def _prefix_wildcard_summarize_exclude_modules(unquantized_layers, quantized_layers):
    """Generate a summarization of the quantization layer configs using prefix wildcards.

    Prefix wildcards means we only consider wildcards that is a prefix with a star in the end.
    We do not consider other wildcards such as: a*b.
    """

    def all_matching_prefix_wildcards(name):
        # include all possible prefix wildcards, and the exact name itself
        wildcards = set([name])
        for i in range(len(name) + 1):
            wildcards.add(name[:i] + "*")
        return wildcards

    def next_formated_matching_prefix_wildcards(name: str) -> Generator[list[str], None, None]:
        """Enumerate formated prefix wildcards. A result may be a combination of prefix wildcards.

        Formated here means we only consider wildcards at dot split. We need two patterns.

        1. a single wildcard: module_name*
        2. a set of 2 wildcards: {module_name, module_name.*}. We need this pattern set because
           module_name* may match other modules with module_name as a prefix.
        """
        for i in range(len(name)):
            if name[i] == ".":
                yield [name[:i] + "*"]
                yield [name[:i], name[:i] + ".*"]
        # in the end, itself only is a wildcard
        yield [name]

    # any of the wildcard in this set cannot be present in the result
    negtive_wild_candidates = set()
    for layer in quantized_layers:
        negtive = all_matching_prefix_wildcards(layer)
        negtive_wild_candidates.update(negtive)
        logger.debug(
            f"Quantized layer {layer}, prefix wildcards {negtive} identified as negative wildcards"
        )

    res_summary = set()
    for layer in unquantized_layers:
        candidate_wildcards = []
        for wildcards in next_formated_matching_prefix_wildcards(layer):
            if any([wildcard in negtive_wild_candidates for wildcard in wildcards]):
                # need a more specific wildcard
                logger.debug(
                    f"Unquantized layer {layer}, prefix wildcards {wildcards} invalidated by negative wildcards"
                )
                continue
            if all([wildcard in res_summary for wildcard in wildcards]):
                # we get covered already, do not need to move forward, and clear candidate
                logger.debug(
                    f"Unquantized layer {layer}, prefix wildcards {wildcards} already covered"
                )
                candidate_wildcards = []
                break
            # find one, now terminate the search
            candidate_wildcards = wildcards
            logger.debug(
                f"Unquantized layer {layer}, prefix wildcards {wildcards} identified as a new match"
            )
            break
        res_summary.update(candidate_wildcards)
    return res_summary


def _detect_exclude_modules(weight_keys: Iterable[str]) -> list[str]:
    quantized_layers = set()
    unquantized_layers = set()

    for key in weight_keys:
        suffix = key.split(".")[-1]

        # Filter kv_cache_scaling_factor.
        if "kv_cache_scaling_factor" in suffix:
            continue

        if "_scaling_factor" in suffix:
            quantized_layers.add(key.rsplit(".", 1)[0])
        else:
            unquantized_layers.add(key.rsplit(".", 1)[0])

    logger.debug(f"pre-unquantized_layers {unquantized_layers}")
    logger.debug(f"pre-quantized_layers {quantized_layers}")

    unquantized_layers = unquantized_layers - quantized_layers

    logger.debug(f"unquantized_layers {unquantized_layers}")
    logger.debug(f"quantized_layers {quantized_layers}")

    res_with_wildcards = _prefix_wildcard_summarize_exclude_modules(
        unquantized_layers, quantized_layers
    )
    return list(res_with_wildcards)


def _get_block_size(model_config: ModelConfig):
    """Return the first block size that is not zero if any."""
    for layer in model_config.layers:
        module_list = [layer.attention.qkv] if layer.attention else []
        if layer.mlp and isinstance(layer.mlp, MLPConfig):
            module_list.extend(
                [
                    layer.mlp.fc,
                    layer.mlp.proj,
                    layer.mlp.gate,
                ]
            )
        if layer.mlp and isinstance(layer.mlp, MOEConfig):
            module_list.extend([layer.mlp.experts.fc, layer.mlp.experts.proj])
        for m in module_list:
            if m is not None and m.awq_block_size != 0:
                return m.awq_block_size
    return 0


def _get_quant_config(
    model_config: ModelConfig,
    weight_keys: Iterable[str],
    kv_cache_dtype: Optional[str] = None,
    pp_size: int = 1,
) -> dict[str, Any]:
    quantization: dict[str, Any] = {"quant_algo": None, "kv_cache_quant_algo": None}

    if model_config.quantization == "fp8":
        quantization.update({"quant_algo": "FP8"})
    elif model_config.quantization == "int4_awq":
        quantization.update(
            {
                "quant_algo": "W4A16_AWQ",
                "group_size": _get_block_size(model_config),
                "has_zero_point": False,
                "pre_quant_scale": True,
            }
        )
    elif model_config.quantization == "w4a8_awq":
        quantization.update(
            {
                "quant_algo": "W4A8_AWQ",
                "group_size": _get_block_size(model_config),
                "has_zero_point": False,
                "pre_quant_scale": True,
            }
        )
    elif model_config.quantization == "int8_sq":
        quantization.update(
            {
                "quant_algo": "W8A8_SQ_PER_CHANNEL",
            }
        )
    elif model_config.quantization == "nvfp4":
        quantization.update(
            {
                "quant_algo": "NVFP4",
                "group_size": _get_block_size(model_config),
            }
        )
    elif model_config.quantization == "nvfp4_awq":
        quantization.update(
            {
                "quant_algo": "NVFP4_AWQ",
                "group_size": _get_block_size(model_config),
                "has_zero_point": False,
                "pre_quant_scale": True,
            }
        )
    elif model_config.quantization == QUANTIZATION_NONE:
        quantization.update(
            {
                "quant_algo": None,
            }
        )
    else:
        quantization.update(
            {
                "quant_algo": model_config.quantization,
            }
        )

    # Deprecate exclude modules for per layer export
    if model_config.quantization != QUANTIZATION_NONE:
        exclude_modules = _detect_exclude_modules(weight_keys)
        # In TRT LLM, the embedding table is shared for the following models, so lm_head quantization format
        # won't be automatically detected in the excluded_modules. We need to manually add it to the exclusions.
        share_embedding_table = model_config.lm_head is None and pp_size == 1
        if share_embedding_table:
            exclude_modules.append("lm_head")
        quantization["exclude_modules"] = exclude_modules

    if kv_cache_dtype is not None:
        quantization.update(
            {
                "kv_cache_quant_algo": kv_cache_dtype,
            }
        )

    return quantization


def convert_to_tensorrt_llm_config(
    model_config: ModelConfig,
    weight_keys: Iterable[str],
    hf_config=None,
):
    """Convert to TensorRT-LLM checkpoint config.

    Args:
        model_config: The model_config to convert.
        weight_keys: The iterable of string of weights exported to the tensorrt_llm checkpoint.
        hf_config: The huggingface model config.
            If provided, we try to use the TensorRT-LLM's export method if available.
    """
    tp_size = model_config.tensor_parallel
    pp_size = model_config.pipeline_parallel

    first_attention_config = None
    first_attention_decoder_config = None
    for decoder_layer in model_config.layers:
        first_attention_config = (
            decoder_layer.attention or decoder_layer.self_attention or decoder_layer.cross_attention
        )

        if first_attention_config is not None:
            first_attention_decoder_config = decoder_layer
            break

    assert first_attention_config is not None and first_attention_decoder_config is not None, (
        "Model must have at least one attention block"
    )

    mapping = {
        "world_size": tp_size * pp_size,
        "tp_size": tp_size,
        "pp_size": pp_size,
    }
    quantization = _get_quant_config(
        model_config, weight_keys, first_attention_config.kv_cache_dtype, pp_size
    )

    decoder_type = model_config.layers[0].decoder_type

    config_architecture = model_config.architecture
    if not config_architecture:
        config_architecture = MODEL_NAME_TO_HF_ARCH_MAP[decoder_type]
    # For Encoder-Decoder model
    is_enc_dec = model_type_is_enc_dec(decoder_type)
    if is_enc_dec:
        # For encoder
        if model_config.enc_dec == "enc":
            config_architecture = MODEL_NAME_TO_HF_ARCH_MAP["enc"]
        # For decoder
        else:
            config_architecture = MODEL_NAME_TO_HF_ARCH_MAP["dec"]

    config = {
        "producer": {
            "name": "modelopt",
            "version": __version__,
        },
        "architecture": config_architecture,
        "dtype": model_config.dtype,
        "logits_dtype": "float16" if model_config.dtype == "bfloat16" else model_config.dtype,
        "num_hidden_layers": len(model_config.layers) * pp_size,
        "num_attention_heads": model_config.num_attention_heads,
        "num_key_value_heads": model_config.num_kv_heads,
        "hidden_size": model_config.hidden_size,
        "norm_epsilon": (
            first_attention_decoder_config.mlp_layernorm.eps
            if is_enc_dec
            else first_attention_decoder_config.input_layernorm.eps
        ),
        "vocab_size": model_config.vocab_size,
        "max_position_embeddings": model_config.max_position_embeddings,
        "hidden_act": model_config.hidden_act,
        "use_parallel_embedding": True,
        "embedding_sharding_dim": 0,
        "head_size": first_attention_decoder_config.attention_head_size,
        "intermediate_size": first_attention_decoder_config.ffn_hidden_size_local * tp_size,
        "position_embedding_type": (
            "alibi"
            if first_attention_decoder_config.use_alibi
            else (
                first_attention_decoder_config.position_embedding_type
                if first_attention_decoder_config.position_embedding_type
                else "rope_gpt_neox"
            )
        ),
        "share_embedding_table": True if (model_config.lm_head is None and pp_size == 1) else False,
        "residual_mlp": first_attention_decoder_config.residual_mlp is not None,
        # Model Optimizer customized fields
        "bias": first_attention_config.dense.bias is not None,
        "rotary_pct": first_attention_decoder_config.rotary_pct,
        "rank": model_config.rank,
        "decoder": first_attention_decoder_config.decoder_type,
        "rmsnorm": _find_layernorm_type(model_config) == LAYERNORM_RMS,
        "lm_head_bias": model_config.lm_head is not None and model_config.lm_head.bias is not None,
    }

    if hf_config is not None:
        try:
            from tensorrt_llm.models import MODEL_MAP
            from tensorrt_llm.models.modeling_utils import Mapping, QuantConfig

            model_cls = MODEL_MAP[model_config.architecture]
            config_cls = getattr(model_cls, "config_class", None)

            if config_cls is not None:
                tensorrt_llm_config = config_cls.from_hugging_face(
                    hf_config_or_dir=hf_config,
                    dtype=model_config.dtype,
                    mapping=Mapping.from_dict(mapping),
                    quant_config=QuantConfig.from_dict(quantization),
                ).to_dict()

                config.update(tensorrt_llm_config)
                # Modelopt uses parallel_embedding by default.
                config.update(
                    {
                        "use_parallel_embedding": True,
                        "embedding_sharding_dim": 0,
                        "logits_dtype": "float16"
                        if model_config.dtype == "bfloat16"
                        else model_config.dtype,
                    }
                )

                return config

        except Exception as e:
            warn(
                "Cannot export tensorrt_llm checkpoint config due to the following error. "
                f"Trying the manual approach: \n{e}"
            )

    if first_attention_decoder_config.rotary_base:
        config["rotary_base"] = first_attention_decoder_config.rotary_base

    if first_attention_decoder_config.rope_scaling:
        config["rotary_scaling"] = first_attention_decoder_config.rope_scaling

    config["mapping"] = mapping
    config["quantization"] = quantization

    layernorm_type_map = {i.name: i.value for i in LayerNormType}
    layernorm_position_map = {i.name: i.value for i in LayerNormPositionType}

    if decoder_type in ["gpt", "gemma", "llama"]:
        pass
    elif decoder_type == "mpt":
        config.update(
            {
                "clip_qkv": first_attention_config.clip_qkv,
                "alibi_bias_max": model_config.layers[0].alibi_bias_max,
            }
        )
    elif decoder_type == "recurrentgemma":
        config["conv_kernel"] = 4
        config["state_size"] = 1
        config["state_dtype"] = "float32"
        config["rnn_hidden_size"] = model_config.layers[0].rnn_hidden_size
        config["rnn_conv_dim_size"] = model_config.layers[0].rnn_hidden_size
        config["logits_soft_cap"] = model_config.layers[0].logits_soft_cap
        config["emb_scale_by_sqrt_dim"] = model_config.layers[0].emb_scale_by_sqrt_dim
        config["layer_types"] = model_config.layers[0].layer_types
    elif is_enc_dec:
        # T5 models use relative position embedding
        # Bart models use learned_absolute position embedding
        config["position_embedding_type"] = (
            "relative" if decoder_type == "t5" else "learned_absolute"
        )
        config["share_embedding_table"] = getattr(model_config, "share_embedding_table")
        config["has_position_embedding"] = (
            False if not getattr(model_config, "position_embedding") else True
        )
        # fallback to RmsNorm if not specified as HF might change class naming for layernorm
        layernorm_type = model_config.layers[0].mlp_layernorm.layernorm_type
        if not layernorm_type:
            layernorm_type = "RmsNorm"
        config["layernorm_type"] = layernorm_type_map[layernorm_type]
        config["has_attention_qkvo_bias"] = (
            True
            if (
                model_config.layers[0].attention.qkv.bias is not None
                if model_config.enc_dec == "enc"
                else model_config.layers[0].self_attention.qkv.bias is not None
            )
            else False
        )
        config["has_mlp_bias"] = False if model_config.layers[0].mlp.fc.bias is None else True
        config["has_model_final_layernorm"] = True if model_config.ln_f else False

        mlp_type_map = {i.name: i.value for i in MLPType}

        config["mlp_type"] = mlp_type_map[
            (
                "GatedMLP"
                if isinstance(model_config.layers[0].mlp, MLPConfig)
                and model_config.layers[0].mlp.gate
                else "MLP"
            )
        ]
        config["use_prompt_tuning"] = False
        config["has_position_embedding"] = False if not model_config.position_embedding else True
        config["has_embedding_layernorm"] = False if not model_config.ln_embed else True
        config["has_embedding_scale"] = False
        config["ffn_hidden_size"] = model_config.layers[0].mlp.fc.weight.shape[0]
        # T5 uses q_scaling to offset attention scaling effect. Bart does not.
        config["q_scaling"] = 1 / config["head_size"] ** 0.5 if decoder_type == "t5" else 1.0
        config["layernorm_position"] = (
            layernorm_position_map["post_layernorm"]
            if decoder_type == "bart"
            else layernorm_position_map["pre_layernorm"]
        )
        config["relative_attention"] = config["position_embedding_type"] == "relative"
        config["max_distance"] = model_config.layers[0].rel_attn_max_distance
        config["num_buckets"] = model_config.layers[0].rel_attn_num_buckets
        config["model_type"] = decoder_type
        config["use_parallel_embedding"] = True
        config["use_implicit_relative_attention"] = False
        if model_config.enc_dec == "dec":
            config["rescale_before_lm_head"] = False
            config["encoder_hidden_size"] = model_config.encoder_hidden_size
            config["encoder_num_heads"] = model_config.encoder_num_heads
            config["encoder_head_size"] = model_config.encoder_head_size
            config["decoder_start_token_id"] = model_config.decoder_start_token_id
            config["eos_token_id"] = model_config.eos_token_id
            config["bos_token_id"] = model_config.bos_token_id
            config["pad_token_id"] = model_config.pad_token_id

    elif decoder_type == "dbrx":
        config["clip_qkv"] = first_attention_decoder_config.clip_qkv

    elif decoder_type == "mllama":
        num_layers = config["num_hidden_layers"]
        num_kv_heads = model_config.num_kv_heads
        cross_attention_layers = set(model_config.layers[0].cross_attention_layers)

        config["num_kv_heads_per_layer"] = [
            0 if i in cross_attention_layers else num_kv_heads for i in range(num_layers)
        ]
        config["num_kv_heads_per_cross_attn_layer"] = [
            num_kv_heads if i in cross_attention_layers else 0 for i in range(num_layers)
        ]

        config["cross_attention"] = True
        config["cross_attention_layers"] = model_config.layers[0].cross_attention_layers
        config["embed_vocab_size"] = model_config.vocab_size + 8
        vision_output_dim = model_config.layers[0].vision_output_dim
        config["vision_output_dim"] = vision_output_dim if vision_output_dim != 0 else 7680
    else:
        raise NotImplementedError(
            f"Cannot export tensorrt_llm checkpoint for model {decoder_type}: {config_architecture}. "
            "It's not supported by TensorRT-LLM yet."
        )

    # For Mixtral and Arctic
    if first_attention_decoder_config.moe_num_experts:
        config["moe"] = {
            "num_experts": first_attention_decoder_config.moe_num_experts,
            "top_k": first_attention_decoder_config.moe_top_k,
            "normalization_mode": 1,  # ExpertScaleNormalizationMode.RENORMALIZE
        }

        if decoder_type == "phi3":
            config["moe"]["normalization_mode"] = 2  # ExpertScaleNormalizationMode.SPARSE_MIXER,
            config["moe"]["sparse_mixer_epsilon"] = (
                first_attention_decoder_config.sparse_mixer_epsilon
            )

        config["mapping"]["moe_tp_size"] = config["mapping"]["tp_size"]
        config["mapping"]["moe_ep_size"] = 1

    # Handle Medusa decoding
    # TODO (chenhany): when inference pp > 1; only last pp has medusa heads
    if model_config.medusa_heads is not None:
        config["base_architecture"] = config["architecture"]
        config["architecture"] = "MedusaForCausalLM"
        # NOTE: max_draft_len is related to the medusa tree len. Currently it is hardcoded to 63.
        config["max_draft_len"] = 63
        config["num_medusa_heads"] = len(model_config.medusa_heads)
        config["num_medusa_layers"] = len(model_config.medusa_heads[0].medusa_layers)

    return config


def prepare_enc_dec_export_dir(tensorrt_llm_config: dict[str, Any], export_root: Path):
    """Prepare the export directory for encoder-decoder model."""
    # For encoder
    if tensorrt_llm_config["architecture"] == "EncoderModel":
        export_dir = export_root.joinpath("encoder")
    # For decoder
    else:
        export_dir = export_root.joinpath("decoder")
    return export_dir


def prepare_t5_decoder_layer(
    layer_config: DecoderLayerConfig,
    model_config: "T5Config",
    enc_dec: str,
    layers: list[DecoderLayerConfig],
):
    """Prepare the config for each decoder layer of encoder-decoder model."""
    layer_config.rel_attn_max_distance = model_config.relative_attention_max_distance
    layer_config.rel_attn_num_buckets = model_config.relative_attention_num_buckets
    if enc_dec == "enc" and layer_config.attention.rel_attn_table is None:
        layer_config.attention.rel_attn_table = layers[0].attention.rel_attn_table
    elif enc_dec == "dec" and layer_config.self_attention.rel_attn_table is None:
        layer_config.self_attention.rel_attn_table = layers[0].self_attention.rel_attn_table
