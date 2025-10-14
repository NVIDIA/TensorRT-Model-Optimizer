# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# MIT License
#
# Copyright (c) 2023 Deep Cognition and Language Research (DeCLaRe) Lab
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

from contextlib import contextmanager
from typing import Any

import torch
import uvloop
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.sampling_params import SamplingParams
from vllm.sequence import IntermediateTensors
from vllm.utils import FlexibleArgumentParser
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput

# from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker

import modelopt.torch.quantization as mtq
from modelopt.torch.utils.dataset_utils import get_dataset_dataloader


@contextmanager
def disable_compilation(model):
    """Context manager to temporarily disable torch.compile"""
    do_not_compile = True
    if hasattr(model, "model"):
        do_not_compile = model.model.do_not_compile
        model.model.do_not_compile = True
    elif hasattr(model, "language_model"):  # VLM requires this
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
    "quant_dataset": "cnn_dailymail",
    "quant_num_samples": 512,
    "quant_format": "NVFP4_DEFAULT_CFG",
    "amax_file_path": None,  # Optional: path to pre-computed amax values (e.g., "/path/to/amax.pt")
}


def fakequant_run_prolog(self):
    tokenizer = AutoTokenizer.from_pretrained(
        self.model_config.tokenizer,
        trust_remote_code=True,
    )
    if tokenizer.pad_token != "<unk>" or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quant_config["amax_file_path"]:
        # If amax file path is provided, we only need to do a simple calibration step
        quant_config["quant_num_samples"] = 1

    calib_dataloader = get_dataset_dataloader(
        dataset_name=quant_config["quant_dataset"],
        tokenizer=tokenizer,
        batch_size=1,
        num_samples=quant_config["quant_num_samples"],
        device=self.device,
    )

    def calibrate_loop(model: Any = None) -> None:
        print("Calibrating model...")
        for batch_idx, batch in tqdm(enumerate(calib_dataloader)):
            input_ids = batch["input_ids"][0]

            # Convert tensor to list of integers for vLLM compatibility
            if torch.is_tensor(input_ids):
                input_ids_list = input_ids.cpu().tolist()
            else:
                input_ids_list = list(input_ids)

            num_groups = len(self.kv_cache_config.kv_cache_groups)
            empty_block_ids = tuple([] for _ in range(num_groups))

            # Build the per-request payload the model runner normally receives from the scheduler.
            req_id = f"req-{batch_idx}"
            new_req = NewRequestData(
                req_id=req_id,
                prompt_token_ids=input_ids_list,
                mm_kwargs=[],
                mm_hashes=[],
                mm_positions=[],
                sampling_params=SamplingParams(max_tokens=1),
                pooling_params=None,
                block_ids=empty_block_ids,
                num_computed_tokens=0,
                lora_request=None,
            )

            # Assemble a SchedulerOutput with all KV-related fields left empty.
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
            intermediate_tensors = None
            forward_pass = scheduler_output.total_num_scheduled_tokens > 0
            if forward_pass and not get_pp_group().is_first_rank:
                intermediate_tensors = IntermediateTensors(
                    get_pp_group().recv_tensor_dict(all_gather_group=get_tp_group())
                )
            self.execute_model(scheduler_output, intermediate_tensors=intermediate_tensors)

    quant_cfg = getattr(mtq, quant_config["quant_format"])

    with disable_compilation(self.model):
        mtq.quantize(self.model, quant_cfg, forward_loop=calibrate_loop)

    # Only print on rank 0 to avoid duplicate output in distributed setups
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(self.model)

    # Override amax values from saved state_dict
    amax_file_path = quant_config["amax_file_path"]
    if amax_file_path:
        print(f"Loading amax values from {amax_file_path}")
        saved_amax_dict = torch.load(amax_file_path, map_location=self.device)
        current_state_dict = self.model.state_dict()

        # Count amax keys in checkpoint and model
        checkpoint_amax_keys = [key for key in saved_amax_dict if key.endswith("amax")]
        model_amax_keys = [key for key in current_state_dict if key.endswith("amax")]

        checkpoint_amax_count = len(checkpoint_amax_keys)
        model_amax_count = len(model_amax_keys)

        # Ensure counts match
        if checkpoint_amax_count != model_amax_count:
            raise ValueError(
                f"Mismatch in amax key counts: checkpoint has {checkpoint_amax_count} "
                f"amax keys but model has {model_amax_count} amax keys. "
            )

        # Update amax values
        for key, value in saved_amax_dict.items():
            if key in current_state_dict:
                current_state_dict[key] = value.to(self.device)

        self.model.load_state_dict(current_state_dict, strict=True)

    mtq.fold_weight(self.model)


# Store the original profile_run method
old_determine_available_memory = Worker.determine_available_memory
old_compile_or_warm_up_model = Worker.compile_or_warm_up_model


# Define new profile_run that includes our modifications
def new_determine_available_memory(self) -> None:
    with disable_compilation(self.model_runner.model):
        results = old_determine_available_memory(self)
    return results


def new_compile_or_warm_up_model(self):
    if quant_config["quant_format"]:
        fakequant_run_prolog(self.model_runner)
    old_compile_or_warm_up_model(self)


# To make sure this monkey patch can be propagated to subprocess,
# Do not put this into functions!
Worker.determine_available_memory = new_determine_available_memory
Worker.compile_or_warm_up_model = new_compile_or_warm_up_model


def main():
    # Create parser that handles both quant and serve arguments
    parser = FlexibleArgumentParser(description="vLLM model server with quantization support")
    parser.add_argument("model", type=str, help="The path or name of the model to serve")
    parser = make_arg_parser(parser)

    # Parse arguments
    args = parser.parse_args()
    # Run the server
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
