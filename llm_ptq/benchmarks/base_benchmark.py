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

import json
import os
import subprocess
import time
from collections import OrderedDict

import tensorrt_llm
import torch
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization import QuantMode


def get_compute_cap():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv"])
    _, csv_value, *_ = output.splitlines()
    return str(int(float(csv_value) * 10))


def get_csv_filename(model, dtype, tp_size, mode, **kwargs):
    sm = get_compute_cap()
    if len(kwargs) == 0:
        kw_pairs = ""
    else:
        kw_pairs = "_" + "_".join([str(k) + str(v) for k, v in kwargs.items()])
    return f"{model}_{dtype}_tp{tp_size}_{mode}{kw_pairs}_sm{sm}.csv"


def get_engine_name(model, dtype, tp_size, rank):
    return "rank{}.engine".format(rank)


def serialize_engine(engine, path):
    logger.info(f"Serializing engine to {path}...")
    tik = time.time()
    with open(path, "wb") as f:
        # engine object is already complies with python buffer protocol, no need to
        # convert it to bytearray before write, converting to bytearray consumes lots of memory
        f.write(engine)
    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    logger.info(f"Engine serialized. Total time: {t}")


class BaseBenchmark(object):
    def __init__(self, engine_dir, model_name, dtype, rank, world_size, serial_build: bool = False):
        self.engine_dir = engine_dir
        self.model_name = model_name
        self.dtype = dtype
        self.runtime_rank = rank
        self.world_size = world_size
        self.engine_model_name = model_name
        self.quant_mode = QuantMode(0)
        self.enable_fp8 = False
        self.mode = ""
        if engine_dir is not None:
            # Read config from engine directory
            config_path = os.path.join(engine_dir, "config.json")
            with open(config_path, "r") as f:
                self.config = json.load(f)
            if "pretrained_config" in self.config:  # new build api branch
                config_dtype = self.config["pretrained_config"]["dtype"]
                assert (
                    dtype == config_dtype
                ), f"Engine dtype ({config_dtype}) != Runtime dtype ({dtype})"
                world_size = self.config["pretrained_config"]["mapping"]["world_size"]
                assert (
                    world_size == self.world_size
                ), f"Engine world size ({world_size}) != Runtime world size ({self.world_size})"
                # Load config into self
                for key, value in self.config["pretrained_config"].items():
                    if key == "ssm_cfg":
                        for ssm_key, ssm_value in value.items():
                            setattr(self, "mamba_" + ssm_key, ssm_value)
                    else:
                        setattr(self, key, value)

                self.quant_mode = QuantMode.from_quant_algo(
                    quant_algo=self.quantization["quant_algo"],  # type: ignore[attr-defined]
                    kv_cache_quant_algo=self.quantization["kv_cache_quant_algo"],  # type: ignore[attr-defined]
                )
                self.enable_fp8 = self.quant_mode.has_fp8_qdq()
                self.fp8_kv_cache = self.quant_mode.has_fp8_kv_cache()

                for key, value in self.config["build_config"].items():
                    setattr(self, key, value)

                for key, value in self.plugin_config.items():  # type: ignore[attr-defined]
                    if "plugin" in key:
                        key = "use_" + key
                    setattr(self, key, value)

                self.engine_name = f"rank{self.runtime_rank}.engine"

                self.num_kv_heads = self.num_key_value_heads  # type: ignore[attr-defined]
                self.num_layers = self.num_hidden_layers  # type: ignore[attr-defined]
                self.num_heads = self.num_attention_heads  # type: ignore[attr-defined]
            else:
                # Read config from engine directory
                config_path = os.path.join(engine_dir, "config.json")
                with open(config_path, "r") as f:
                    self.config = json.load(f)
                # Sanity checks
                config_dtype = self.config["builder_config"]["precision"]
                assert (
                    dtype == config_dtype
                ), f"Engine dtype ({config_dtype}) != Runtime dtype ({dtype})"
                world_size = self.config["builder_config"]["tensor_parallel"]
                assert (
                    world_size == self.world_size
                ), f"Engine world size ({world_size}) != Runtime world size ({self.world_size})"
                # Load config into self
                for key, value in self.config["builder_config"].items():
                    if key == "quant_mode":
                        self.quant_mode = QuantMode(value)
                    elif key in "name":
                        self.engine_model_name = value
                    else:
                        setattr(self, key, value)
                self.enable_fp8 = self.quant_mode.has_fp8_qdq()
                self.fp8_kv_cache = self.quant_mode.has_fp8_kv_cache()
                for key, value in self.config["plugin_config"].items():
                    # Same effect as self.use_foo_plugin = config.json["foo_plugin"]
                    if "plugin" in key:
                        key = "use_" + key
                    setattr(self, key, value)
                self.engine_name = get_engine_name(
                    self.engine_model_name, self.dtype, self.world_size, self.runtime_rank
                )
        else:
            self.engine_name = get_engine_name(
                self.engine_model_name, self.dtype, self.world_size, self.runtime_rank
            )

        self.runtime_mapping = tensorrt_llm.Mapping(
            world_size=self.world_size, rank=self.runtime_rank, tp_size=self.world_size
        )
        if not serial_build:
            torch.cuda.set_device(self.runtime_rank % self.runtime_mapping.gpus_per_node)

        self.csv_filename = ""  # lazy init

    def get_report_dict(self, benchmark_profiler=None):
        report_fields = [
            "batch_size",
            "input_length",
            "output_length",
            "gpu_peak_mem(gb)",
            "build_time(s)",
            "tokens_per_sec",
            "percentile95(ms)",
            "percentile99(ms)",
            "latency(ms)",
            "compute_cap",
        ]
        report_dict = OrderedDict.fromkeys(report_fields)
        report_dict["model_name"] = self.model_name
        report_dict["world_size"] = self.world_size
        report_dict["precision"] = self.dtype
        report_dict["quantization"] = str(self.quant_mode)
        report_dict["compute_cap"] = "sm" + get_compute_cap()
        return report_dict

    def get_csv_filename(self):
        if len(self.csv_filename) == 0:
            self.csv_filename = get_csv_filename(
                self.model_name,
                self.dtype,
                self.world_size,
                self.mode,
                fp8linear=int(self.enable_fp8),
            )
        return self.csv_filename

    def print_report_header(self, csv=False, benchmark_profiler=None):
        if csv and self.runtime_rank == 0:
            report_dict = self.get_report_dict(benchmark_profiler)
            line = ",".join(report_dict.keys())
            print(line)
            with open(self.get_csv_filename(), "a") as file:
                file.write(line + "\n")

    def get_config(self):
        raise NotImplementedError

    def prepare_inputs(self, config):
        raise NotImplementedError

    def run(self, inputs, config, benchmark_profiler=None):
        raise NotImplementedError

    def report(self, config, latency):
        raise NotImplementedError
