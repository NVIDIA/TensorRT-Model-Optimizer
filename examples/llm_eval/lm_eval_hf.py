# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/tree/aa457edc3d64d81530159cd3a182932320c78f8c

# MIT License
#
# Copyright (c) 2020 EleutherAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
import warnings

from lm_eval import utils
from lm_eval.__main__ import cli_evaluate, parse_eval_args, setup_parser
from lm_eval.api.model import T
from lm_eval.models.huggingface import HFLM
from quantization_utils import quantize_model

import modelopt.torch.opt as mto
from modelopt.torch.quantization.utils import is_quantized


def create_from_arg_obj(cls: type[T], arg_dict: dict, additional_config: dict | None = None) -> T:
    """Overrides the HFLM.create_from_arg_obj"""

    quant_cfg = arg_dict.pop("quant_cfg", None)
    auto_quantize_bits = arg_dict.pop("auto_quantize_bits", None)
    calib_batch_size = arg_dict.pop("calib_batch_size", None)
    calib_size = arg_dict.pop("calib_size", 512)
    compress = arg_dict.pop("compress", False)

    additional_config = {} if additional_config is None else additional_config
    additional_config = {k: v for k, v in additional_config.items() if v is not None}

    # Enable automatic save/load of modelopt state huggingface checkpointing
    mto.enable_huggingface_checkpointing()

    model_obj = cls(**arg_dict, **additional_config)
    model_obj.tokenizer.padding_side = "left"
    if is_quantized(model_obj.model):
        # return if model is already quantized
        warnings.warn("Skipping quantization: model is already quantized.")
        return model_obj

    if quant_cfg:
        if not calib_batch_size:
            calib_batch_size = model_obj.batch_size

        quantize_model(
            model=model_obj,
            quant_cfg=quant_cfg.split(",") if auto_quantize_bits is not None else quant_cfg,
            tokenizer=model_obj.tokenizer,
            batch_size=calib_batch_size,
            calib_size=calib_size,
            auto_quantize_bits=auto_quantize_bits,
            test_generated=False,
            compress=compress,
        )

    return model_obj


HFLM.create_from_arg_obj = classmethod(create_from_arg_obj)


def setup_parser_with_modelopt_args():
    parser = setup_parser()
    parser.add_argument(
        "--quant_cfg",
        type=str,
        help=(
            "Quantization format. If `--auto_quantize_bits` is specified, this argument specifies the "
            "comma-separated list of quantization quantization formats that will be searched by `auto_quantize`"
        ),
    )
    parser.add_argument(
        "--auto_quantize_bits",
        type=float,
        help=(
            "Effective bits constraint for auto_quantize. If not set, "
            "regular quantization will be applied."
        ),
    )
    parser.add_argument(
        "--calib_batch_size", type=int, help="Batch size for quantization calibration"
    )
    parser.add_argument(
        "--calib_size", type=int, help="Calibration size for quantization", default=512
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress the model after quantization",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser_with_modelopt_args()
    args = parse_eval_args(parser)
    model_args = utils.simple_parse_args_string(args.model_args)

    if args.trust_remote_code:
        import datasets

        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
        model_args["trust_remote_code"] = True
        args.trust_remote_code = None

    model_args.update(
        {
            "quant_cfg": args.quant_cfg,
            "auto_quantize_bits": args.auto_quantize_bits,
            "calib_batch_size": args.calib_batch_size,
            "calib_size": args.calib_size,
            "compress": args.compress,
        }
    )

    args.model_args = model_args

    cli_evaluate(args)
