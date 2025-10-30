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
from contextlib import contextmanager
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.worker.gpu_worker import Worker as BaseWorker

import modelopt.torch.quantization as mtq
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader


@contextmanager
def disable_compilation(model):
    print("fakequant_worker.disable_compilation")
    do_not_compile = True
    if hasattr(model, "model"):
        do_not_compile = model.model.do_not_compile
        model.model.do_not_compile = True
    elif hasattr(model, "language_model"):
        do_not_compile = model.language_model.model.do_not_compile
        model.language_model.model.do_not_compile = True
    else:
        raise ValueError("Model does not have a model or language_model attribute")

    try:
        yield
    finally:
        if hasattr(model, "model"):
            model.model.do_not_compile = do_not_compile
        elif hasattr(model, "language_model"):
            model.language_model.model.do_not_compile = do_not_compile


quant_config: dict[str, Any] = {
    "quant_dataset": "magpie",
    "quant_num_samples": 512,
    "quant_format": "NVFP4_DEFAULT_CFG",
    "amax_file_path": None,
}


def _fakequant_run_prolog_worker(self) -> None:
    print("fakequant_worker._fakequant_run_prolog")
    tokenizer = AutoTokenizer.from_pretrained(
        self.model_runner.model_config.tokenizer,
        trust_remote_code=True,
    )
    if tokenizer.pad_token != "<unk>" or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quant_config["amax_file_path"]:
        print("Will load amax, so only do a single sample calibration")
        quant_config["quant_num_samples"] = 1

    calib_dataloader = get_dataset_dataloader(
        dataset_name=quant_config["quant_dataset"],
        tokenizer=tokenizer,
        batch_size=1,
        num_samples=quant_config["quant_num_samples"],
        device=self.device,
    )

    def calibrate_loop(model: Any = None) -> None:
        for batch_idx, batch in tqdm(enumerate(calib_dataloader)):
            input_ids = batch["input_ids"][0]

            # Convert tensor to list of integers for vLLM compatibility
            if torch.is_tensor(input_ids):
                input_ids_list = input_ids.cpu().tolist()
            else:
                input_ids_list = list(input_ids)

            num_groups = len(self.model_runner.kv_cache_config.kv_cache_groups)
            empty_block_ids = tuple([] for _ in range(num_groups))

            req_id = f"req-{batch_idx}"
            new_req = NewRequestData(
                req_id=req_id,
                prompt_token_ids=input_ids_list,
                # mm_kwargs=[],
                # mm_hashes=[],
                # mm_positions=[],
                mm_features=[],
                sampling_params=SamplingParams(max_tokens=1),
                pooling_params=None,
                block_ids=empty_block_ids,
                num_computed_tokens=0,
                lora_request=None,
            )

            scheduler_output = SchedulerOutput(
                scheduled_new_reqs=[new_req],
                scheduled_cached_reqs=CachedRequestData.make_empty(),
                num_scheduled_tokens={req_id: len(input_ids_list)},
                total_num_scheduled_tokens=len(input_ids_list),
                scheduled_spec_decode_tokens={},
                scheduled_encoder_inputs={},
                num_common_prefix_blocks=[0] * num_groups,
                finished_req_ids=set(),
                free_encoder_mm_hashes=[],
                structured_output_request_ids={},
                grammar_bitmask=None,
                kv_connector_metadata=None,
            )
            self.execute_model(scheduler_output)

    quant_cfg = getattr(mtq, quant_config["quant_format"])

    with disable_compilation(self.model_runner.model):
        print("quantizing model...")
        mtq.quantize(self.model_runner.model, quant_cfg, forward_loop=calibrate_loop)

    amax_file_path = quant_config["amax_file_path"]
    if amax_file_path:
        print(f"Loading amax values from {amax_file_path}")
        saved_amax_dict = torch.load(amax_file_path, map_location=self.model_runner.device)
        current_state_dict = self.model_runner.model.state_dict()

        # Count amax keys in checkpoint and model
        checkpoint_amax_keys = [key for key in saved_amax_dict if key.endswith("_amax")]
        model_amax_keys = [key for key in current_state_dict if key.endswith("_amax")]
        for key in checkpoint_amax_keys:
            if key not in model_amax_keys:
                print(f"Key {key} not found in model state dict, but exists in checkpoint")
        for key in model_amax_keys:
            if key not in checkpoint_amax_keys:
                raise ValueError(
                    f"Key {key} not found in checkpoint state dict, but exists in model"
                )

        checkpoint_amax_count = len(checkpoint_amax_keys)
        model_amax_count = len(model_amax_keys)

        # Ensure counts match
        if checkpoint_amax_count != model_amax_count:
            warnings.warn(
                f"Mismatch in amax key counts: checkpoint has {checkpoint_amax_count} "
                f"amax keys but model has {model_amax_count} amax keys. This can happen if the model is using PP."
            )

        # Update amax values
        for key, value in saved_amax_dict.items():
            if key in current_state_dict:
                current_state_dict[key] = value.to(current_state_dict[key].device)

        self.model_runner.model.load_state_dict(current_state_dict)
        torch.distributed.barrier()

    if amax_file_path is None:
        # Sync amax across TP can be done here if needed
        pass
        # for name, buffer in self.model_runner.model.named_buffers():
        #     if name.endswith("_amax"):
        #         print("syncing amax across TP for", name)
        #         torch.distributed.all_reduce(
        #             buffer, op=torch.distributed.ReduceOp.MAX, group=get_tp_group().device_group
        #         )
        # torch.distributed.barrier()

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        mtq.print_quant_summary(self.model_runner.model)

    mtq.fold_weight(self.model_runner.model)
    for name, module in self.model_runner.model.named_modules():
        if name.endswith("weight_quantizer"):
            assert not module.is_enabled, f"quantizer {name} is still enabled"


class FakeQuantWorker(BaseWorker):
    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        with disable_compilation(self.model_runner.model):
            return super().determine_available_memory()

    def compile_or_warm_up_model(self) -> None:
        if quant_config["quant_format"]:
            _fakequant_run_prolog_worker(self)
        super().compile_or_warm_up_model()
