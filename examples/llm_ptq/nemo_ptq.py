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

import copy
import os
import time

import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from megatron.core import parallel_state
from megatron.core.transformer.module import Float16Module
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.core.config import hydra_runner
from nemo.utils.model_utils import load_config, save_artifacts, unwrap_model
from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_tensorrt_llm_checkpoint
from modelopt.torch.utils import print_rank_0
from modelopt.torch.utils.dataset_utils import _CustomDataset

mp.set_start_method("spawn", force=True)


def get_dataset(data="cnn_dailymail"):
    if data == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
        text_column = "text"
    elif data == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
        text_column = "article"
    return dataset, text_column


def get_calib_dataloader(
    data="cnn_dailymail", batch_size=64, calib_size=512, max_sequence_length=512
):
    dataset, text_column = get_dataset(data)
    calib_size = max(min(len(dataset), calib_size), batch_size)
    for i in range(calib_size // batch_size):
        batch = dataset[i * batch_size : (i + 1) * batch_size][text_column]
        for j in range(len(batch)):
            batch[j] = batch[j][:max_sequence_length]
        yield batch


def get_dataloader_for_fwd_bwd(
    data="cnn_dailymail", tokenizer=None, batch_size=1, calib_size=512, sequence_length=512
):
    dataset, text_column = get_dataset(data)
    encodings = {k: [] for k in ["tokens", "labels", "loss_mask", "position_ids", "attention_mask"]}
    for i, data in zip(range(calib_size), dataset):
        tokens = tokenizer.text_to_ids(data[text_column])
        tokens, labels = tokens[:-1], tokens[1:]
        loss_mask = [1.0] * len(tokens)
        attention_mask = torch.tril(torch.ones((sequence_length, sequence_length))).unsqueeze(0)

        if len(tokens) < sequence_length:
            num_tokens = len(tokens)
            tokens = tokens + [tokenizer.pad_id] * (sequence_length - num_tokens)
            labels = labels + [tokenizer.pad_id] * (sequence_length - num_tokens)
            loss_mask = loss_mask + [0.0] * (sequence_length - num_tokens)
            attention_mask[:, num_tokens:] = 0.0
        elif len(tokens) > sequence_length:
            tokens = tokens[:sequence_length]
            labels = labels[:sequence_length]
            loss_mask = loss_mask[:sequence_length]

        attention_mask = attention_mask < 0.5
        encodings["tokens"].append(torch.tensor(tokens).cuda())
        encodings["labels"].append(torch.tensor(labels).cuda())
        encodings["loss_mask"].append(torch.tensor(loss_mask).cuda())
        encodings["position_ids"].append(torch.arange(sequence_length, dtype=torch.int64).cuda())
        encodings["attention_mask"].append(attention_mask.cuda())

    dataset = _CustomDataset(encodings)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


QUANT_CFG_CHOICES = {
    "int8": "INT8_DEFAULT_CFG",
    "int8_sq": "INT8_SMOOTHQUANT_CFG",
    "fp8": "FP8_DEFAULT_CFG",
    "int4_awq": "INT4_AWQ_CFG",
    "w4a8_awq": "W4A8_AWQ_BETA_CFG",
}


@hydra_runner(
    config_path="config",
    config_name="megatron_quantization",
)
def main(cfg) -> None:
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is required for the inference.")

    # dtype is used for non-quantized layers
    supported_dtype = ["fp16", "bf16"]
    assert cfg.export.dtype in supported_dtype, (
        f"{cfg.export.dtype} not supported. Supported dtypes are {supported_dtype}"
    )
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
        if cfg.quantization.auto_quantize_bits is not None:
            # Enable activation checkpointing if auto_quantize is enabled to reduce memory footprint
            model_cfg.activations_checkpoint_granularity = "full"
            model_cfg.activations_checkpoint_method = "uniform"
            model_cfg.activations_checkpoint_num_layers = 1
            # `forward_step` for auto_quantize is called with a single batch; Hence set number of micro_batches to 1;
            model_cfg.global_batch_size = cfg.inference.batch_size
            model_cfg.micro_batch_size = cfg.inference.batch_size

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

    if cfg.quantization.algorithm and cfg.quantization.algorithm != "null":
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

        dataloader = [data for data in dataloader]

        def forward_loop(model):
            print("Calibrating the model...")
            for i, batch in enumerate(tqdm(dataloader)):
                model.predict_step(batch, i)

        # =================== Start Quantization ====================

        # Always turn on FP8 kv cache to save memory footprint.
        # For int8_sq, we do not quantize kv cache to preserve accuracy.
        # TODO: Investigate why enabling FP8 kv cache will cause accuracy regressions for nemotron.
        enable_quant_kv_cache = (
            "int8" not in cfg.quantization.algorithm
            and cfg.export.decoder_type != "gpt"
            and not cfg.quantization.disable_kv_cache
        )
        print(f"{'Enable' if enable_quant_kv_cache else 'Disable'} KV cache quantization")

        start_time = time.time()
        if cfg.quantization.auto_quantize_bits is not None:
            # Check if list of quantization formats provided for auto quantize search are supported
            qformat_list = cfg.quantization.algorithm.split(",")
            assert all(qformat in QUANT_CFG_CHOICES.keys() for qformat in qformat_list), (
                "One or more quantization formats provided for auto quantize search are not supported"
            )

            dataloader = get_dataloader_for_fwd_bwd(
                cfg.quantization.calib_dataset,
                model.tokenizer,
                cfg.inference.batch_size,
                cfg.quantization.num_calib_size,
            )
            model, search_state = mtq.auto_quantize(
                model,
                data_loader=dataloader,
                constraints={"effective_bits": float(cfg.quantization.auto_quantize_bits)},
                quantization_formats=[QUANT_CFG_CHOICES[format] for format in qformat_list]
                + [None],
                forward_step=lambda model, data: model.fwd_bwd_step(
                    iter([data]), forward_only=True
                ),
                forward_backward_step=lambda model, data: model.fwd_bwd_step(
                    iter([data]), forward_only=False
                ),
                num_calib_steps=len(dataloader),
                # Limit the number of score steps to avoid long auto-quantize time
                num_score_steps=min(len(dataloader), 128 // cfg.inference.batch_size),
                verbose=True,
            )

            # Disable activation checkpointing
            model._reset_activation_checkpointing_args()

            # KV cache is not quantized during auto_quantize; So lets quantize and calibrate just KV cache now
            if enable_quant_kv_cache:
                mtq.set_quantizer_by_cfg(
                    model,
                    quant_cfg={
                        "*output_quantizer": {"num_bits": (4, 3), "axis": None, "enable": True}
                    },
                )
                # Lets calibrate only the output quantizer this time
                with mtq.set_quantizer_by_cfg_context(
                    model, {"*": {"enable": False}, "*output_quantizer": {"enable": True}}
                ):
                    mtq.calibrate(model, algorithm="max", forward_loop=forward_loop)

        else:
            # Check if quantization.algorithm is in QUANT_CFG_CHOICES
            assert cfg.quantization.algorithm in QUANT_CFG_CHOICES, (
                f"Quantization format {cfg.quantization.algorithm} not supported"
            )
            atq_config = getattr(mtq, QUANT_CFG_CHOICES[cfg.quantization.algorithm])

            if "awq" in cfg.quantization.algorithm:
                atq_config = copy.deepcopy(
                    getattr(mtq, QUANT_CFG_CHOICES[cfg.quantization.algorithm])
                )
                weight_quantizer = atq_config["quant_cfg"]["*weight_quantizer"]
                if isinstance(weight_quantizer, list):
                    weight_quantizer = weight_quantizer[0]
                weight_quantizer["block_sizes"][-1] = cfg.quantization.awq_block_size

            atq_config["quant_cfg"]["*output_quantizer"] = {
                "num_bits": 8 if cfg.quantization.algorithm == "int8_sq" else (4, 3),
                "axis": None,
                "enable": enable_quant_kv_cache,
            }

            model = mtq.quantize(model, atq_config, forward_loop)

        end_time = time.time()
        tot_time = end_time - start_time
        tput = cfg.quantization.num_calib_size / tot_time
        print_rank_0(f"Quantization done. Total time used {tot_time}s. Throughput {tput} samples/s")
        # =================== End Quantization ======================

        if cfg.export.decoder_type == "gpt":
            # We found squared_relu may have an under-calibration problem.
            # Clamp the scaling_factor with a min threshold to avoid under-calibration.
            for name, module in model.named_modules():
                # Clamping scaling_factor is performed for fp8 and int8_sq
                if (
                    name.endswith(".input_quantizer")
                    and module.amax is not None
                    and module.num_bits in [8, (4, 3)]
                ):
                    module.amax = torch.clamp(module.amax, min=0.01 * module.maxbound)

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
