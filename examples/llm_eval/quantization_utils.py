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


import torch
from transformers import AutoTokenizer

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import need_calibration
from modelopt.torch.quantization.plugins import register_hf_attentions_on_the_fly
from modelopt.torch.utils.dataset_utils import (
    create_forward_loop,
    get_dataset_dataloader,
    get_max_batch_size,
)

MAX_SEQ_LEN = 4096
MAX_OUTPUT_LEN = 512

# This is an example to customize the quantization config.
# Modify your custom config for debugging or research purposes.
CUSTOM_CONFIG = {
    "MY_QUANT_CONFIG": {
        "quant_cfg": {
            "*weight_quantizer": {"num_bits": 4, "block_sizes": {-1: 128}, "enable": True},
            "*input_quantizer": {"num_bits": 8, "type": "dynamic", "block_sizes": {-1: None}},
            # Disable sensitive layers such as `lm_head`, gate layers in MoE etc.
            **mtq.config._default_disabled_quantizer_cfg,
        },
        "algorithm": "max",
    },
}


def get_tokenizer(ckpt_path, max_seq_len=MAX_SEQ_LEN, trust_remote_code=False):
    """Returns the tokenizer from the model ckpt_path."""
    print(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path,
        model_max_length=max_seq_len,
        padding_side="left",
        trust_remote_code=trust_remote_code,
    )
    if type(tokenizer).__name__ == "QWenTokenizer":
        # qwen use token id 151643 as pad and eos tokens
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(151643)

    # can't set attribute 'pad_token' for "<unk>"
    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def _quantize_model_with_dataset(
    lm,
    quant_cfg: str | list[str],
    calib_dataset,
    auto_quantize_bits=None,
    batch_size=1,
    compress=False,
):
    if hasattr(lm, "gpt2"):
        net = lm.gpt2
    elif hasattr(lm, "model"):
        net = lm.model
    else:
        net = lm

    if auto_quantize_bits is not None:
        quant_cfg_for_search = [
            getattr(mtq, quant_fmt) for quant_fmt in quant_cfg if quant_fmt != "NONE"
        ]

        def loss_func(output, data):
            # For transformers AutoModelForCausalLM models, the outputs are wrapped in `CausalLMOutputWithPast`
            # which contains the loss attribute.
            return output.loss

        net, _ = mtq.auto_quantize(
            net,
            constraints={"effective_bits": auto_quantize_bits},
            quantization_formats=quant_cfg_for_search,
            data_loader=calib_dataset,
            forward_step=lambda model, batch: model(**batch),
            loss_func=loss_func,
            num_calib_steps=len(calib_dataset),
            num_score_steps=min(
                len(calib_dataset), 128 // batch_size
            ),  # Limit the number of score steps to avoid long calibration time
            verbose=True,
        )
    else:
        mtq_cfg = CUSTOM_CONFIG.get(quant_cfg)  # type: ignore [arg-type]
        if mtq_cfg is None:
            mtq_cfg = getattr(mtq, quant_cfg)  # type: ignore [arg-type]

        calibrate_loop = None
        use_calibration = need_calibration(mtq_cfg)
        if not use_calibration:
            print("Dynamic quantization. Calibration skipped.")
        else:
            # The calibrate_loop is a custom defined method to run the model with the input data.
            # The basic version looks like:
            #
            # def calibrate_loop(model, dataloader):
            #     for data in dataloader:
            #         model(**data)
            #
            # We also provided a util method to generate the forward_loop with additional error handlings.
            calibrate_loop = create_forward_loop(dataloader=calib_dataset)

        quantize_bmm_attention = False
        for key in mtq_cfg["quant_cfg"]:
            if "bmm_quantizer" in key:
                quantize_bmm_attention = True
        if quantize_bmm_attention:
            register_hf_attentions_on_the_fly(net)

        net = mtq.quantize(net, mtq_cfg, calibrate_loop)
    mtq.print_quant_summary(net)
    # Compress or fold weights for faster evaluation.
    if compress:
        mtq.compress(net)
    else:
        mtq.fold_weight(net)


def quantize_model(
    model,
    quant_cfg: str | list[str],
    tokenizer,
    batch_size,
    calib_size,
    auto_quantize_bits=None,
    data="cnn_dailymail",
    test_generated=True,
    compress=False,
):
    """Quantizes the model with the provided calibration dataset.

    Args:
        model: the model to be quantized.
        quant_cfg: the quantization algorithm config name if simple quantization is used.
                   the list of quantization algorithm config names if auto quantization is used.
        tokenizer: the tokenizer.
        batch_size: the calibration batch size for each calibration inference run.
        calib_size: the total calibration dataset size.
        auto_quantize_bits: The effective bits constraint for auto_quantize.
        data: the name of the calibration dataset.
        test_generated:  If ``True``, test the generated text before and after quantization.
        compress: If ``True``, compress the model after quantization.
    """
    if "AWQ" in quant_cfg:
        print(
            "\n####\nAWQ calibration could take longer than other calibration methods. "
            "Consider reducing calib_size to reduce calibration time.\n####\n"
        )

    device = model.device
    if hasattr(model, "model"):
        device = model.model.device

    if batch_size == 0:
        if auto_quantize_bits is not None or torch.distributed.is_initialized():
            raise ValueError("We dont support automatic batch size inference for this case.")

        net = model.gpt2 if hasattr(model, "gpt2") else model.model

        # We let the system to determine the max data batch for each forward.
        batch_size = get_max_batch_size(net)
        print(f"Update calib batch {batch_size}")

    calib_dataloader = get_dataset_dataloader(
        dataset_name=data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_samples=calib_size,
        device=device,
        include_labels=auto_quantize_bits is not None,
    )

    if test_generated:
        input_str = tokenizer.decode(next(iter(calib_dataloader))["input_ids"][0])
        generated_str_before_ptq = model.run(input_str)

    _quantize_model_with_dataset(
        model, quant_cfg, calib_dataloader, auto_quantize_bits, batch_size, compress
    )

    if test_generated:
        generated_str_after_ptq = model.run(input_str)

        print("--------")
        print(f"example test input: {input_str}")
        print("--------")
        print(f"example outputs before ptq: {generated_str_before_ptq}")
        print("--------")
        print(f"example outputs after ptq: {generated_str_after_ptq}")
