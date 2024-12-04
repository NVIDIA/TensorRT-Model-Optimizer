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

import sys
from pathlib import Path
from typing import List, Union

from transformers import AutoTokenizer

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.plugins import register_hf_attentions_on_the_fly

sys.path.append(str(Path(__file__).resolve().parent / "../common"))
from dataset_utils import create_forward_loop, get_dataset_dataloader, get_max_batch_size

MAX_SEQ_LEN = 2048
MAX_OUTPUT_LEN = 512


def get_tokenizer(ckpt_path, max_seq_len=MAX_SEQ_LEN):
    """Returns the tokenizer from the model ckpt_path."""
    print(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path, model_max_length=max_seq_len, padding_side="left", trust_remote_code=True
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
    lm, quant_cfg: Union[str, List[str]], calib_dataset, auto_quantize_bits=None, batch_size=1
):
    if hasattr(lm, "gpt2"):
        net = lm.gpt2
    elif hasattr(lm, "model"):
        net = lm.model
    else:
        net = lm

    if auto_quantize_bits is not None:
        quant_cfg_for_search = [
            quant_fmt if quant_fmt != "NONE" else None for quant_fmt in quant_cfg
        ]
        net, _ = mtq.auto_quantize(
            net,
            constraints={"effective_bits": auto_quantize_bits},
            quantization_formats=quant_cfg_for_search,
            data_loader=calib_dataset,
            forward_step=lambda model, batch: model(**batch),
            loss_func=lambda output, data: output.loss,
            num_calib_steps=len(calib_dataset),
            num_score_steps=min(
                len(calib_dataset), 128 // batch_size
            ),  # Limit the number of score steps to avoid long calibration time
            verbose=True,
        )
    else:
        atq_cfg = getattr(mtq, quant_cfg)  # type: ignore [arg-type]

        # The calibrate_loop is a custom defined method to run the model with the input data.
        # The basic version looks like:
        #
        # def calibrate_loop(model, dataloader):
        #     for data in dataloader:
        #         model(**data)
        #
        # We also provided a util method to generate the forward_loop with additional error handlings.
        calibrate_loop = create_forward_loop(dataloader=calib_dataset)

        def is_dynamic(atq_cfg):
            def _not_dynamic(cfg):
                return (
                    cfg.get("enable", True)
                    and cfg.get("type", "") != "dynamic"
                    and cfg.get("*", {}).get("enable", True)
                )

            for _, cfg in atq_cfg.get("quant_cfg", {}).items():
                # quantization like W4A8 has a list of weight quantizers
                if isinstance(cfg, list):
                    for config in cfg:
                        if _not_dynamic(config):
                            return False
                else:
                    if _not_dynamic(cfg):
                        return False

            return True

        use_calibration = not is_dynamic(atq_cfg)
        if not use_calibration:
            print("Dynamic quantization. Calibration skipped.")

        quantize_bmm_attention = False
        for key in atq_cfg["quant_cfg"]:
            if "bmm_quantizer" in key:
                quantize_bmm_attention = True
        if quantize_bmm_attention:
            register_hf_attentions_on_the_fly(net)

        net = mtq.quantize(net, atq_cfg, calibrate_loop if use_calibration else None)
    mtq.print_quant_summary(net)
    # Fold weights for faster evaluation.
    mtq.fold_weight(net)


def quantize_model(
    model,
    quant_cfg: str,
    tokenizer,
    batch_size,
    calib_size,
    auto_quantize_bits=None,
    data="cnn_dailymail",
    test_generated=True,
):
    """Quantizes the model with the provided calibration dataset.

    Args:
        model: the model to be quantized.
        quant_cfg: the quantization algorithm config name.
        tokenizer: the tokenizer.
        batch_size: the calibration batch size for each calibration inference run.
        calib_size: the total calibration dataset size.
        auto_quantize_bits: The effective bits constraint for AutoQuantize.
        data: the name of the calibration dataset.
        test_generated:  If ``True``, test the generated text before and after quantization.
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
        assert auto_quantize_bits is None, "AutoQuantize requires batch_size to be set."

        if hasattr(model, "gpt2"):
            net = model.gpt2
        else:
            net = model.model

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

    _quantize_model_with_dataset(model, quant_cfg, calib_dataloader, auto_quantize_bits, batch_size)

    if test_generated:
        generated_str_after_ptq = model.run(input_str)

        print("--------")
        print(f"example test input: {input_str}")
        print("--------")
        print(f"example outputs before ptq: {generated_str_before_ptq}")
        print("--------")
        print(f"example outputs after ptq: {generated_str_after_ptq}")
