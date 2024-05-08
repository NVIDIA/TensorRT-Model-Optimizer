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

import os
from math import ceil

import tensorrt_llm
import torch
from model_runner import read_config
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.profiler import bytes_to_target_unit

from base_benchmark import BaseBenchmark  # isort:skip


def element_size(dtype: str):
    str_to_size_in_bytes = dict(
        float16=2, float32=4, int64=8, int32=4, int8=1, bool=1, bfloat16=2, fp8=1
    )
    return str_to_size_in_bytes[dtype]


class GPTBenchmark(BaseBenchmark):
    def __init__(self, args, batch_sizes, in_out_lens, rank, world_size):
        # Model Optimizer modification
        model_config = read_config(args.engine_dir, rank)
        args.model = model_config.model_name

        super().__init__(
            args.engine_dir, args.model, args.dtype, rank, world_size, args.serial_build
        )
        self.batch_sizes = batch_sizes
        self.in_out_lens = in_out_lens
        self.num_beams = args.num_beams
        self.mode = args.mode
        self.build_time = 0

        self.cuda_graph_mode = args.enable_cuda_graph
        self.build_config = None
        # this dtype may be modified based on quantization mode later, when the fp8/int8 kv cache is used
        self.kv_dtype = args.dtype

        # approximate the weights size in the engine by using engine size
        # the actual weights size shall be smaller because there are some other data in the engine file.
        # for large model, this approximate is close enough.
        self.weights_size_approx = 0

        if args.engine_dir is not None:
            # Get build configs from engine directory is done in base class
            # Deserialize engine from engine directory
            self.serialize_path = os.path.join(args.engine_dir, self.engine_name)
            with open(self.serialize_path, "rb") as f:
                engine_buffer = f.read()
                self.weights_size_approx = len(engine_buffer)
        else:
            # Model Optimizer Modification
            raise NotImplementedError("Please specify engine_dr")

        assert engine_buffer is not None
        if args.build_only:
            return

        if args.model == "chatglm_6b":
            self.sampling_config = tensorrt_llm.runtime.SamplingConfig(
                end_id=130005,
                pad_id=3,
                num_beams=self.num_beams,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            self.decoder = tensorrt_llm.runtime.ChatGLMGenerationSession(
                model_config, engine_buffer, self.runtime_mapping
            )
        elif args.model in ["chatglm2_6b", "chatglm3_6b"]:
            self.sampling_config = tensorrt_llm.runtime.SamplingConfig(
                end_id=2, pad_id=0, num_beams=self.num_beams, top_k=args.top_k, top_p=args.top_p
            )
            self.decoder = tensorrt_llm.runtime.GenerationSession(
                model_config, engine_buffer, self.runtime_mapping
            )
        elif "mamba" in args.model:
            model_config.mamba_d_state = self.mamba_d_state
            model_config.mamba_d_conv = self.mamba_d_conv
            model_config.mamba_expand = self.mamba_expand
            self.sampling_config = tensorrt_llm.runtime.SamplingConfig(
                end_id=0, pad_id=0, top_k=args.top_k, top_p=args.top_p
            )
            self.decoder = tensorrt_llm.runtime.MambaLMHeadModelGenerationSession(
                model_config,
                engine_buffer,
                self.runtime_mapping,
                cuda_graph_mode=self.cuda_graph_mode,
            )
        else:
            end_id = 50256
            pad_id = 50256
            if "llama" in args.model:
                end_id = 2
                pad_id = 0
            self.sampling_config = tensorrt_llm.runtime.SamplingConfig(
                end_id=end_id,
                pad_id=pad_id,
                num_beams=self.num_beams,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            self.decoder = tensorrt_llm.runtime.GenerationSession(
                model_config,
                engine_buffer,
                self.runtime_mapping,
                cuda_graph_mode=self.cuda_graph_mode,
            )

    def get_config(self):
        for inlen, outlen in self.in_out_lens:
            for batch_size in self.batch_sizes:
                yield (batch_size, inlen, outlen)

    def prepare_inputs(self, config):
        batch_size, inlen, outlen = config[0], config[1], config[2]
        input_ids = torch.randint(100, (batch_size, inlen)).int().cuda()
        input_lengths = torch.tensor([inlen for _ in range(batch_size)]).int().cuda()

        self.decoder.setup(batch_size, inlen, outlen, beam_width=self.num_beams)
        return (input_ids, input_lengths)

    def get_report_dict(self, benchmark_profiler=None):
        report_dict = super().get_report_dict(benchmark_profiler=benchmark_profiler)
        if benchmark_profiler is not None:
            report_dict["generation_time(ms)"] = None
            report_dict["total_generated_tokens"] = None
            report_dict["generation_tokens_per_second"] = None
        return report_dict

    def run(self, inputs, config, benchmark_profiler=None):
        batch_size, inlen, outlen = config[0], config[1], config[2]
        self.decoder.setup(batch_size, inlen, outlen, beam_width=self.num_beams)
        if self.remove_input_padding:
            self.decoder.decode_batch(
                inputs[0], self.sampling_config, benchmark_profiler=benchmark_profiler
            )
        else:
            self.decoder.decode(
                inputs[0], inputs[1], self.sampling_config, benchmark_profiler=benchmark_profiler
            )
        torch.cuda.synchronize()

    @staticmethod
    def kv_cache_elem_per_token(config: BuildConfig, tp_size, pp_size) -> int:
        # you need to multiply the size by element size, and multiply by the seq length
        # Warning: this function returns the upper bound between different ranks when any one of the following is true:
        # num_layer % pp_size !=0, hidden_size % num_kv_heads != 0, num_kv_heads % tp_size != 0
        local_nlayers = ceil(config.num_layers / pp_size)
        kv_heads = config.num_kv_heads if config.num_kv_heads is not None else config.num_heads
        size_per_head = ceil(config.hidden_size / kv_heads)
        local_heads = ceil(kv_heads / tp_size)
        return 2 * local_nlayers * size_per_head * local_heads

    def check_memory(self, io_shapes: list, raise_exception=False):
        """Compare the estimated GPU memory requirements for weights + activations + kv cache
        with the total GPU memory and log it.
        Raise exception when the \\p raise_exception parameter is true.
        """
        # we don't want to block the test due to this
        if self.build_config is None:
            tensorrt_llm.logger.warning(
                "Didn't have the build config object, skipping check the memory"
            )
            return
        assert isinstance(self.build_config, BuildConfig)
        batch_size, inlen, outlen = io_shapes[0], io_shapes[1], io_shapes[2]
        kv_cache_size_in_bytes = (
            batch_size
            * self.num_beams
            * (inlen + outlen)
            * self.kv_cache_elem_per_token(
                self.build_config, self.runtime_mapping.tp_size, self.runtime_mapping.pp_size
            )
            * element_size(self.kv_dtype)
        )
        # when MHA is OOTB, it requires 2x KV cache size, one for past as engine input, one for present as engine output
        if not self.use_gpt_attention_plugin:
            kv_cache_size_in_bytes *= 2
        kv_cache_size_in_mb = bytes_to_target_unit(kv_cache_size_in_bytes, "MiB")
        activation_size_in_mb = bytes_to_target_unit(
            self.decoder.runtime.engine.device_memory_size, "MiB"
        )
        weights_size_in_mb = bytes_to_target_unit(self.weights_size_approx, "MiB")
        total_memory_approx_in_mb = kv_cache_size_in_mb + activation_size_in_mb + weights_size_in_mb
        _, _, total = tensorrt_llm.profiler.device_memory_info()
        total_in_mb = bytes_to_target_unit(total, "MiB")
        prefix = "[Memory Estimation]"

        mem_msg = (
            f"{prefix} activation memory:{activation_size_in_mb:.3f} MiB,"
            f" kv_cache:{kv_cache_size_in_mb:.3f} MiB, weights"
            f" approximate:{weights_size_in_mb:.3f} MiB, approximate required GPU memory:"
            f" {total_memory_approx_in_mb:.3f} MiB, total GPU memory: {total_in_mb:.3f} MiB"
        )
        tensorrt_llm.logger.info(mem_msg)

        build_args = dict(
            batch_size=batch_size,
            num_beams=self.num_beams,
            input_length=inlen,
            output_length=outlen,
            max_batch_size=self.build_config.max_batch_size,
            max_input_len=self.build_config.max_input_len,
            max_output_len=self.build_config.max_output_len,
            max_beam_width=self.build_config.max_beam_width,
        )
        for k, v in build_args.items():
            tensorrt_llm.logger.info(f"{prefix} {k}:{v}")

        tensorrt_llm.logger.info(
            'grep the "Total Activation" and "Total Weights" from verbose TRT engine build log to'
            " see the precise memory size for those."
        )
        if raise_exception and total_memory_approx_in_mb >= total_in_mb:
            raise Exception(
                "Total memory estimation bigger than total gpu memory, the case will likely to OOM,"
                " needs enhancement of waive the test case, see logs about the memory usage details"
            )

    def report(
        self,
        config,
        latency,
        percentile95,
        percentile99,
        peak_gpu_used,
        csv,
        benchmark_profiler=None,
    ):
        report_dict = super().get_report_dict()
        batch_size, inlen, outlen = config[0], config[1], config[2]
        tokens_per_sec = round(batch_size * outlen / (latency / 1000), 2)
        report_dict["batch_size"] = batch_size
        report_dict["input_length"] = inlen
        report_dict["output_length"] = outlen
        report_dict["latency(ms)"] = latency
        report_dict["build_time(s)"] = self.build_time
        report_dict["tokens_per_sec"] = tokens_per_sec
        report_dict["percentile95(ms)"] = percentile95
        report_dict["percentile99(ms)"] = percentile99
        report_dict["gpu_peak_mem(gb)"] = peak_gpu_used
        if benchmark_profiler is not None:
            iter_count = benchmark_profiler.get_aux_info("iter_count")
            generation_time_ms = benchmark_profiler.get_timer_value("generation_time")
            generation_step_count = benchmark_profiler.get_aux_info("generation_step_count")
            token_per_step = batch_size * self.num_beams
            total_tokens = generation_step_count * token_per_step
            report_dict["generation_time(ms)"] = round(generation_time_ms / iter_count, 3)
            report_dict["total_generated_tokens"] = total_tokens / iter_count
            tokens_per_second = round(total_tokens * 1000.0 / generation_time_ms, 3)
            report_dict["generation_tokens_per_second"] = tokens_per_second

        if self.runtime_rank == 0:
            if csv:
                line = ",".join([str(v) for v in report_dict.values()])
                print(line)
                with open(self.get_csv_filename(), "a") as file:
                    file.write(line + "\n")
            else:
                kv_pairs = [f"{k} {v}" for k, v in report_dict.items()]
                line = "[BENCHMARK] " + " ".join(kv_pairs)
                print(line)
