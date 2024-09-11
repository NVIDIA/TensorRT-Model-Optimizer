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

"""An example to convert an Model Optimizer exported model to tensorrt_llm."""

import argparse
import subprocess
from pathlib import Path
from typing import Optional, Union

import tensorrt_llm
import torch
from run_tensorrt_llm import run
from tensorrt_llm.models import PretrainedConfig
from transformers import AutoTokenizer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def build_tensorrt_llm(
    pretrained_config: Union[str, Path],
    engine_dir: Union[str, Path],
    max_input_len: int = 200,
    max_output_len: int = 200,
    max_batch_size: int = 1,
    max_beam_width: int = 1,
    max_num_tokens: Optional[int] = None,
    num_build_workers: int = 1,
    enable_sparsity: bool = False,
    max_prompt_embedding_table_size: int = 0,
    gather_context_logits: bool = False,
):
    """The API to convert the TensorRT-LLM checkpoint to engines.

    Args:
        pretrained_config: The pretrained_config (file path) exported by
            ``modelopt.torch.export.export_tensorrt_llm_checkpoint``.
        engine_dir: The target output directory to save the built tensorrt_llm engines.
        max_input_len: The max input sequence length.
        max_output_len: The max output sequence length.
        max_batch_size: The max batch size.
        max_beam_width: The max beam search width.
        max_num_tokens: The max number of tokens that can be processed at the same time.
            For the context phase, the max_num_tokens counts the full sequence length.
            For the generation phase, the max_num_tokens counts only the ones under generation
            as the input sequence has been processed as cached.
            max_num_tokens should fall between [max_batch_size * max_beam_width, max_batch_size * max_input_len].
            when inflight batching is enabled.
            Higher max_num_tokens means more GPU memory will be used for resource allocation.
            If not specified the max_num_tokens will be set to the max bound.
            Details: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/perf_best_practices.md
        num_build_workers: The number of workers to use for the building process.
            If build time is a concern, you can increase this worker count to num of GPUs.
            At a lost of higer CPU memory usage footprint.
            If CPU memory is limited, num_build_workers should be set to 1 to conserve memory.
        enable_sparsity: The switch to enable sparsity for TRT compiler.
            With this flag, the TRT compiler will search tactics of sparse kernels for each node of which
            weight tensors are sparsified. This increases engine building time significantly.
        max_prompt_embedding_table_size: Length of the prepended/concatenated embeddings (either multimodal
            feature embeddings or prompt tuning embeddings) to the LLM input embeddings.
        gather_context_logits: Whether context logits can be returned as a part of the outputs.
    """
    engine_dir = Path(engine_dir)
    engine_dir.mkdir(parents=True, exist_ok=True)

    pretrained_config_path = Path(pretrained_config)
    assert pretrained_config_path.exists()
    config = PretrainedConfig.from_json_file(pretrained_config_path)
    ckpt_dir = pretrained_config_path.parent

    timing_cache_file = (
        torch.cuda.get_device_name().replace(" ", "_")
        + "_trtllm_"
        + tensorrt_llm.__version__
        + ".cache"
    )
    timing_cache_path = engine_dir / timing_cache_file

    log_level = "warning"

    if max_batch_size < 4:
        print(
            "Warning: TensorRT LLM may hit a runtime issue with batch size is smaller than 4 on some models."
            " Force set to 4"
        )
        max_batch_size = 4

    is_no_quant_or_fp8 = config.quantization.quant_algo in [
        "FP8",
        None,
    ]

    use_fused_mlp = is_no_quant_or_fp8 and config.hidden_act in [
        "silu",
        "swiglu",
        "fast-swiglu",
        "gelu",
        "geglu",
    ]

    if config.quantization.exclude_modules:
        for module_name in config.quantization.exclude_modules:
            # fp8_context_fhma requires all attention.dense to be quantized
            if "attention.dense" in module_name:
                is_no_quant_or_fp8 = False
            # For AutoQuant, fc and gate might not be quantized at the same time
            # TODO: relax this limitation on the TRT-LLM side
            if "gate" in module_name or "fc" in module_name:
                use_fused_mlp = False

    quant_algo = config.quantization.quant_algo
    use_qdq = quant_algo in ["FP8", "W8A8_SQ_PER_CHANNEL"]

    builder_opt = 4 if "RecurrentGemma" not in config.architecture else 0

    speculative_decoding_mode = "medusa" if "Medusa" in config.architecture else None

    build_cmd = "trtllm-build "
    build_cmd += f"--checkpoint_dir {ckpt_dir} "
    build_cmd += f"--input_timing_cache {timing_cache_path} "
    build_cmd += f"--output_timing_cache {timing_cache_path} "
    build_cmd += f"--log_level {log_level} "
    build_cmd += f"--output_dir {engine_dir} "
    build_cmd += f"--workers {num_build_workers} "
    build_cmd += f"--max_batch_size {max_batch_size} "
    build_cmd += f"--max_input_len {max_input_len} "
    build_cmd += f"--max_seq_len {max_output_len + max_input_len} "
    build_cmd += f"--max_beam_width {max_beam_width} "
    build_cmd += f"--max_prompt_embedding_table_size {max_prompt_embedding_table_size} "
    build_cmd += f"--builder_opt {builder_opt} "
    build_cmd += f"--max_num_tokens {max_batch_size * max_input_len} "
    build_cmd += (
        "--reduce_fusion enable "
        if config.mapping.pp_size == 1
        and config.architecture
        not in ["DbrxForCausalLM", "BaichuanForCausalLM", "QWenForCausalLM", "GPTForCausalLM"]
        else ""
    )

    print(
        "Hint: For max througput, please build the engine with --multiple_profiles enable flag. "
        "This is not enabled by default to save engine build time."
    )

    if use_fused_mlp:
        build_cmd += "--use_fused_mlp " if "RecurrentGemma" not in config.architecture else ""
    if enable_sparsity:
        build_cmd += "--weight_sparsity "

    if not use_qdq:
        build_cmd += "--gemm_plugin auto "

    if max_num_tokens:
        build_cmd += f"--max_num_tokens {max_num_tokens} "

    if speculative_decoding_mode:
        build_cmd += f"--speculative_decoding_mode {speculative_decoding_mode} "

    if is_no_quant_or_fp8:
        build_cmd += "--use_paged_context_fmha enable "

    if gather_context_logits:
        build_cmd += "--gather_context_logits "  # for evaluation benchmarking purpose

    print("trtllm-build command:")
    print(f"{build_cmd}")

    subprocess.run(build_cmd, shell=True, check=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        tokenizer.save_pretrained(engine_dir)
    except Exception as e:
        print(f"Cannot copy tokenizer to the engine dir. {e}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default="")
    parser.add_argument("--max_output_len", type=int, default=512)
    parser.add_argument("--max_input_len", type=int, default=2048)
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--max_num_beams", type=int, default=1)
    parser.add_argument("--engine_dir", type=str, default="/tmp/modelopt")
    parser.add_argument("--tokenizer", type=str, default="")
    parser.add_argument(
        "--input_texts",
        type=str,
        default=(
            "Born in north-east France, Soyer trained as a|Born in California, Soyer trained as a"
        ),
        help="Input texts. Please use | to separate different batches.",
    )
    parser.add_argument("--num_build_workers", type=int, default="1")
    parser.add_argument("--enable_sparsity", type=str2bool, default=False)
    parser.add_argument(
        "--max_prompt_embedding_table_size",
        "--max_multimodal_len",
        type=int,
        default=0,
        help="Setting to a value > 0 enables support for prompt tuning or multimodal input.",
    )

    return parser.parse_args()


def main(args):
    build_tensorrt_llm(
        pretrained_config=args.model_config,
        engine_dir=args.engine_dir,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        max_batch_size=args.max_batch_size,
        max_beam_width=args.max_num_beams,
        num_build_workers=args.num_build_workers,
        enable_sparsity=args.enable_sparsity,
        max_prompt_embedding_table_size=args.max_prompt_embedding_table_size,
    )

    if args.model_config is not None and all(
        model_name not in args.model_config for model_name in ("vila", "llava")
    ):
        # Reduce output_len for the inference run example.
        args.max_output_len = 100

        run(args)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
