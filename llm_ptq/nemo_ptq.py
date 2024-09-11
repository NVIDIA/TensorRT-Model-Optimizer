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

import copy
import os
import time

import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from megatron.core import parallel_state
from megatron.core.transformer.module import Float16Module
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.transformer.text_generation import (
    LengthParam,
)
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.core.config import hydra_runner
from nemo.utils.model_utils import load_config, save_artifacts, unwrap_model
from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer
from tqdm import tqdm

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_tensorrt_llm_checkpoint
from modelopt.torch.utils import print_rank_0

mp.set_start_method("spawn", force=True)


def get_calib_dataloader(
    data="cnn_dailymail", batch_size=64, calib_size=512, max_sequence_length=512
):
    if data == "pileval":
        dataset = load_dataset(
            "json", data_files="https://the-eye.eu/public/AI/pile/val.jsonl.zst", split="train"
        )
        text_column = "text"
    elif data == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
        text_column = "text"
    elif data == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
        text_column = "article"
    calib_size = max(min(len(dataset), calib_size), batch_size)
    for i in range(calib_size // batch_size):
        batch = dataset[i * batch_size : (i + 1) * batch_size][text_column]
        for j in range(len(batch)):
            batch[j] = batch[j][:max_sequence_length]
        yield batch


QUANT_CFG_CHOICES = {
    "int8": mtq.INT8_DEFAULT_CFG,
    "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
    "fp8": mtq.FP8_DEFAULT_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
}


@hydra_runner(config_path="config", config_name="megatron_quantization")
def main(cfg) -> None:
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is required for the inference.")

    # dtype is used for non-quantized layers
    supported_dtype = ["fp16", "bf16"]
    assert (
        cfg.export.dtype in supported_dtype
    ), f"{cfg.export.dtype} not supported. Supported dtypes are {supported_dtype}"
    torch_dtype = torch.bfloat16 if cfg.export.dtype == "bf16" else torch.float16

    model_cfg = load_config(cfg.model_file)

    with open_dict(model_cfg):
        model_cfg.activations_checkpoint_method = None
        model_cfg.activations_checkpoint_granularity = None
        model_cfg.tensor_model_parallel_size = cfg.tensor_model_parallel_size
        model_cfg.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
        model_cfg.sequence_parallel = False
        # Only custom Model Optimizer spec is supported for PTQ: this custom spec is largely based on local Megatron-LM
        # layer definitions to avoid Transformer Engine implementations that are currently not supported.
        model_cfg.name = "modelopt"

    # trainer required for restoring model parallel models
    trainer = Trainer(strategy=NLPDDPStrategy(), **cfg.trainer)
    connector = NLPSaveRestoreConnector()

    model = MegatronGPTModel.restore_from(
        restore_path=cfg.model_file,
        trainer=trainer,
        override_config_path=model_cfg,
        save_restore_connector=connector,
    )
    model.freeze()

    print_rank_0(model)
    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.module.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    # Check whether the DDP is initialized
    if parallel_state.is_unitialized():

        def dummy():
            return

        if model.trainer.strategy.launcher is not None:
            model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
        model.trainer.strategy.setup_environment()

    config = OmegaConf.to_container(cfg.inference)
    model.set_inference_config(config)

    if cfg.quantization.algorithm and cfg.quantization.algorithm in QUANT_CFG_CHOICES:
        if "awq" in cfg.quantization.algorithm:
            if cfg.quantization.num_calib_size > 32:
                print_rank_0(
                    "AWQ calibration could take longer with calib_size ="
                    f" {cfg.quantization.num_calib_size}, Using calib_size=32 instead"
                )
                cfg.quantization.num_calib_size = 32
            print_rank_0(
                "\nAWQ calibration could take longer than other calibration methods. Please"
                " increase the batch size to speed up the calibration process. Batch size can be"
                " set by adding the argument inference.batch_size=<batch_size> to the command"
                " line.\n"
            )

        dataloader = get_calib_dataloader(
            cfg.quantization.calib_dataset,
            cfg.inference.batch_size,
            cfg.quantization.num_calib_size,
            cfg.inference.max_context_length,
        )
        # =================== Start Quantization ====================
        atq_config = QUANT_CFG_CHOICES[cfg.quantization.algorithm]

        if "awq" in cfg.quantization.algorithm:
            atq_config = copy.deepcopy(QUANT_CFG_CHOICES[cfg.quantization.algorithm])
            weight_quantizer = atq_config["quant_cfg"]["*weight_quantizer"]  # type: ignore
            if isinstance(weight_quantizer, list):
                weight_quantizer = weight_quantizer[0]
            weight_quantizer["block_sizes"][-1] = cfg.quantization.awq_block_size

        # Always turn on FP8 kv cache to save memory footprint.
        # For int8_sq, we do not quantize kv cache to preserve accuracy.
        # TODO: Investigate why enabling FP8 kv cache will cause accuracy regressions for nemotron.
        enable_quant_kv_cache = (
            "int8" not in cfg.quantization.algorithm and cfg.export.decoder_type != "gptnext"
        )
        print(f'{"Enable" if enable_quant_kv_cache else "Disable"} KV cache quantization')
        atq_config["quant_cfg"]["*output_quantizer"] = {  # type: ignore[index]
            "num_bits": 8 if cfg.quantization.algorithm == "int8_sq" else (4, 3),
            "axis": None,
            "enable": enable_quant_kv_cache,
        }

        dataloader = [data for data in dataloader]

        def forward_loop(model):
            print("Calibrating the model...")
            for i, batch in enumerate(tqdm(dataloader)):
                model.predict_step(batch, i)

        start_time = time.time()
        model = mtq.quantize(model, atq_config, forward_loop)  # type: ignore[arg-type]
        end_time = time.time()
        tot_time = end_time - start_time
        tput = cfg.quantization.num_calib_size / tot_time
        print_rank_0(f"Quantization done. Total time used {tot_time}s. Throughput {tput} samples/s")
        # =================== End Quantization ======================

        if cfg.export.decoder_type == "gptnext":
            # We found squared_relu may have an under-calibration problem.
            # Clamp the scaling_factor with a min threshold to avoid under-calibration.
            maxbound = 0
            if cfg.quantization.algorithm == "fp8":
                maxbound = 448
            elif cfg.quantization.algorithm == "int8_sq":
                maxbound = 127
            model = mtq.postprocess_amax(
                model, "*input_quantizer", lambda amax: torch.clamp(amax, min=0.01 * maxbound)
            )

        if torch.distributed.get_rank() == 0:
            mtq.print_quant_summary(model)

    length_params: LengthParam = {
        "max_length": 100,
        "min_length": 100,
    }

    response = model.generate(
        inputs=[
            "Born in north-east France, Soyer trained as a",
            "Born in California, Soyer trained as a",
        ],
        length_params=length_params,
    )

    if torch.distributed.get_rank() == 0:
        print(f'Example NeMo output after PTQ: {response["sentences"]}"')

    if model_cfg.megatron_amp_O2:
        model.model = unwrap_model(model.model, Float16Module)

    export_path = cfg.export.get("path", os.getcwd())

    start_time = time.time()
    export_tensorrt_llm_checkpoint(
        model,
        cfg.export.decoder_type,
        torch_dtype,
        export_dir=export_path,
        inference_tensor_parallel=cfg.export.inference_tensor_parallel,
        inference_pipeline_parallel=cfg.export.inference_pipeline_parallel,
        use_nfs_workspace=cfg.trainer.num_nodes > 1,
    )
    end_time = time.time()
    print_rank_0(
        f"Model config exported to: {export_path}. Total time used {end_time - start_time}s"
    )
    if torch.distributed.get_rank() == 0:
        save_artifacts(model, export_path, use_abspath=True)


if __name__ == "__main__":
    main()
