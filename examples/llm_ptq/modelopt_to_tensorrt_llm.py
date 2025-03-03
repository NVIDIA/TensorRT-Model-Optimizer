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

"""An example to convert an Model Optimizer exported model to tensorrt_llm."""

import argparse
import subprocess
import warnings
from pathlib import Path
from typing import Optional, Union

import tensorrt_llm
import torch
from packaging.version import parse
from tensorrt_llm.llmapi import BuildConfig
from tensorrt_llm.models import PretrainedConfig
from transformers import AutoTokenizer

try:
    # run depends on features from the min-supported TensorRT-LLM
    from run_tensorrt_llm import run
except Exception as e:
    warnings.warn(f"Cannot run TensorRT-LLM inference: {e}")
    run = None


MIN_TENSORRT_LLM_VERSION = "0.13.0"


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
    max_prompt_embedding_table_size: int = BuildConfig.max_prompt_embedding_table_size,
    max_encoder_input_len: int = BuildConfig.max_encoder_input_len,
    perf_mode: bool = False,
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
            Details: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.html
        num_build_workers: The number of workers to use for the building process.
            If build time is a concern, you can increase this worker count to num of GPUs.
            At a lost of higer CPU memory usage footprint.
            If CPU memory is limited, num_build_workers should be set to 1 to conserve memory.
        enable_sparsity: The switch to enable sparsity for TRT compiler.
            With this flag, the TRT compiler will search tactics of sparse kernels for each node of which
            weight tensors are sparsified. This increases engine building time significantly.
        max_prompt_embedding_table_size: Length of the prepended/concatenated embeddings (either multimodal
            feature embeddings or prompt tuning embeddings) to the LLM input embeddings.
        max_encoder_input_len: Maximum encoder input length for enc-dec models.
        perf_mode: Whether build the engine with max perf at a cost of longer build time and less flexibility.
        checkpoint_format: The model checkpoint format. Choose between [tensorrt_llm, hf].
        tp: tensor_parallel_size. Effective for hf checkpoint_format only.
    """
    engine_dir = Path(engine_dir)
    engine_dir.mkdir(parents=True, exist_ok=True)

    pretrained_config_path = Path(pretrained_config)
    assert pretrained_config_path.exists()
    ckpt_dir = pretrained_config_path.parent

    timing_cache_file = (
        torch.cuda.get_device_name().replace(" ", "_")
        + "_trtllm_"
        + tensorrt_llm.__version__
        + ".cache"
    )
    timing_cache_path = engine_dir / timing_cache_file

    if not max_num_tokens:
        # tensorrt-llm recommends max max_num_tokens to be 16384
        max_num_tokens = min(max_batch_size * max_input_len, 16384)

    config = PretrainedConfig.from_json_file(pretrained_config_path)

    log_level = "warning"

    use_paged_context_fmha = config.quantization.quant_algo in [
        "FP8",
        "W4A8_AWQ",
        "NVFP4",
        None,
    ]

    use_fused_mlp = "RecurrentGemma" not in config.architecture
    if config.quantization.exclude_modules:
        for module_name in config.quantization.exclude_modules:
            # fp8_context_fhma requires all attention.dense to be quantized
            if "attention.dense" in module_name:
                use_paged_context_fmha = False
            # For AutoQuant, fc and gate might not be quantized at the same time
            # TODO: relax this limitation on the TRT-LLM side
            if "gate" in module_name or "fc" in module_name:
                use_fused_mlp = False

    quant_algo = config.quantization.quant_algo
    use_qdq = quant_algo in ["FP8", "W8A8_SQ_PER_CHANNEL"]

    speculative_decoding_mode = "medusa" if "Medusa" in config.architecture else None

    if num_build_workers > torch.cuda.device_count():
        num_build_workers = torch.cuda.device_count()
        print(f"Cap num_build_workers to num gpus: ${num_build_workers}")

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
    build_cmd += f"--max_encoder_input_len {max_encoder_input_len} "
    build_cmd += (
        "--reduce_fusion enable "
        if config.mapping.pp_size == 1
        and config.architecture
        not in [
            "DbrxForCausalLM",
            "BaichuanForCausalLM",
            "QWenForCausalLM",
            "GPTForCausalLM",
        ]
        else ""
    )

    if use_fused_mlp:
        build_cmd += "--use_fused_mlp enable "
    else:
        build_cmd += "--use_fused_mlp disable "

    if enable_sparsity:
        build_cmd += "--weight_sparsity "

    # Low batch size scenario
    if max_batch_size <= 4 and quant_algo == "FP8":
        build_cmd += "--gemm_plugin fp8 "
    if quant_algo == "NVFP4":
        build_cmd += "--gemm_plugin nvfp4 "
    elif not use_qdq:
        build_cmd += "--gemm_plugin auto "

    build_cmd += f"--max_num_tokens {max_num_tokens} "

    if speculative_decoding_mode:
        build_cmd += f"--speculative_decoding_mode {speculative_decoding_mode} "

    if use_paged_context_fmha:
        build_cmd += "--use_paged_context_fmha enable "

    if perf_mode:
        build_cmd += "--multiple_profiles enable"
    elif not speculative_decoding_mode:
        build_cmd += "--gather_context_logits "  # for evaluation benchmarking purpose

    print(f"trtllm-build command:\n{build_cmd}")

    assert parse(tensorrt_llm.__version__) >= parse(MIN_TENSORRT_LLM_VERSION), (
        f"Detected lower version of tensorrt_llm installed instead of {MIN_TENSORRT_LLM_VERSION}. "
        f"Please build the tensorrt_llm engines with tensorrt_llm version {MIN_TENSORRT_LLM_VERSION} "
        " or higher instead.\n\n Build command: {build_cmd}"
    )
    subprocess.run(build_cmd, shell=True, check=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        tokenizer.save_pretrained(engine_dir)
    except Exception as e:
        warnings.warn(f"Cannot copy tokenizer to the engine dir. {e}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default="")
    parser.add_argument("--max_output_len", type=int, default=512)
    parser.add_argument("--max_input_len", type=int, default=2048)
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--max_num_beams", type=int, default=1)
    parser.add_argument(
        "--perf",
        action="store_true",
        help="Build engines for max perf benchmark",
    )
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
        default=BuildConfig.max_prompt_embedding_table_size,
        help="Maximum prompt embedding table size for prompt tuning, "
        "or maximum multimodal input size for multimodal models.",
    )
    parser.add_argument(
        "--max_encoder_input_len",
        type=int,
        default=BuildConfig.max_encoder_input_len,
        help="Maximum encoder input length for enc-dec models.",
    )
    parser.add_argument(
        "--trust_remote_code",
        help="Set trust_remote_code for Huggingface models and tokenizers",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--skip_run",
        help="Skip the inference run",
        default=False,
        action="store_true",
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
        max_encoder_input_len=args.max_encoder_input_len,
        perf_mode=args.perf,
    )

    if (
        args.model_config is not None
        and all(model_name not in args.model_config for model_name in ("vila", "llava"))
        and run is not None
    ):
        # Reduce output_len for the inference run example.
        args.max_output_len = 100

        if not args.skip_run:
            run(args)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
