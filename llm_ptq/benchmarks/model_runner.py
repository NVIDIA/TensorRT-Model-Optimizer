# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from tensorrt_llm.builder import Engine
from tensorrt_llm.runtime import ModelConfig


def read_config(engine_dir, rank) -> ModelConfig:
    # the new engine format
    engine = Engine.from_dir(engine_dir, rank)
    pretrained_config = engine.config.pretrained_config
    build_config = engine.config.build_config

    tp_size = pretrained_config.mapping.tp_size
    num_heads = pretrained_config.num_attention_heads // tp_size
    num_kv_heads = pretrained_config.num_key_value_heads
    num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size
    hidden_size = pretrained_config.hidden_size // tp_size
    head_size = pretrained_config.head_size

    rnn_config_items = [
        "conv_kernel",
        "layer_types",
        "rnn_hidden_size",
        "state_size",
        "state_dtype",
    ]
    rnn_configs_kwargs = {}
    for item in rnn_config_items:
        if hasattr(pretrained_config, item):
            rnn_configs_kwargs[item] = getattr(pretrained_config, item)

    return ModelConfig(
        max_batch_size=build_config.max_batch_size,
        max_beam_width=build_config.max_beam_width,
        vocab_size=pretrained_config.vocab_size,
        num_layers=pretrained_config.num_hidden_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        head_size=head_size,
        gpt_attention_plugin=bool(build_config.plugin_config.gpt_attention_plugin),
        mamba_conv1d_plugin=bool(build_config.plugin_config.mamba_conv1d_plugin),
        remove_input_padding=build_config.plugin_config.remove_input_padding,
        paged_kv_cache=build_config.plugin_config.paged_kv_cache,
        paged_state=build_config.plugin_config.paged_state,
        tokens_per_block=build_config.plugin_config.tokens_per_block,
        quant_mode=pretrained_config.quant_mode,
        gather_context_logits=build_config.gather_context_logits,
        gather_generation_logits=build_config.gather_generation_logits,
        dtype=pretrained_config.dtype,
        max_prompt_embedding_table_size=build_config.max_prompt_embedding_table_size,
        lora_plugin=build_config.plugin_config.lora_plugin,
        lora_target_modules=build_config.lora_config.lora_target_modules,
        trtllm_modules_to_hf_modules=build_config.lora_config.trtllm_modules_to_hf_modules,
        max_medusa_tokens=(
            pretrained_config.max_draft_len if hasattr(pretrained_config, "max_draft_len") else 0
        ),
        num_medusa_heads=(
            pretrained_config.num_medusa_heads
            if hasattr(pretrained_config, "num_medusa_heads")
            else 0
        ),
        use_custom_all_reduce=build_config.plugin_config.use_custom_all_reduce,
        moe_tp_mode=(
            pretrained_config.moe_tp_mode if hasattr(pretrained_config, "moe_tp_mode") else 0
        ),
        **rnn_configs_kwargs,
        gpu_weights_percent=1,
    )
