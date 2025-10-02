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

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from distill_trainer import DRAFT_VOCAB_SIZE, MODEL_PATH, EagleTPTrainer
from eagle_utils import DataCollatorWithPadding, make_eagle_supervised_data_module
from transformers import AutoTokenizer

from modelopt.torch.speculative.config import EAGLE3_DEFAULT_CFG

# Hyperparameters for profiling
torch.manual_seed(0)
INPUT_LENGTH = 1024
# DRAFT_VOCAB_SIZE = 128256
# DRAFT_VOCAB_SIZE = 32000
# MODEL_PATH = "/home/scratch.omniml_data_1/models_ci/meta-llama/Llama-3.1-8B-Instruct"
# MODEL_PATH = "/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_modelopt/hf-local/meta-llama/Llama-3.2-1B-Instruct"
# MODEL_PATH ="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# MODEL_PATH = "openai/gpt-oss-20b"
# MODEL_PATH = "/home/scratch.omniml_data_1/models_ci/meta-llama/Llama-3.3-70B-Instruct"


def _setup_distributed(rank, args, backend="nccl"):
    """Initialize distributed environment"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.master_port
    os.environ["LOCAL_RANK"] = str(rank)
    # Initialize process group
    dist.init_process_group(backend, rank=rank, world_size=args.world_size)
    if rank in args.student_ranks:
        torch.cuda.set_device(args.student_devices[rank])
    else:
        torch.cuda.set_device(args.teacher_devices[rank - len(args.student_ranks)])
    print(
        f"Starting process rank={rank}, device={torch.cuda.current_device()}, world_size={args.world_size}"
    )


def train(rank, args):
    _setup_distributed(rank, args)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, model_max_length=INPUT_LENGTH)
    data_module = make_eagle_supervised_data_module(tokenizer, args, use_offline_training=False)

    train_dataloader = torch.utils.data.DataLoader(
        data_module["train_dataset"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=DataCollatorWithPadding(max_length=INPUT_LENGTH),
        drop_last=True,
    )

    trainer_cls = {
        "tp": EagleTPTrainer,
        # "mp": EagleMPTrainer,
    }[args.teacher_parallel]

    distill_metadata = {
        "base_model_hidden_states": (
            torch.Size([int(args.batch_size / len(args.student_ranks)), INPUT_LENGTH, 2048]),
            torch.bfloat16,
        ),
        "aux_hidden_states": (
            torch.Size([int(args.batch_size / len(args.student_ranks)), INPUT_LENGTH, 2048 * 3]),
            torch.bfloat16,
        ),
        "base_model_logits": (
            torch.Size(
                [int(args.batch_size / len(args.student_ranks)), INPUT_LENGTH, DRAFT_VOCAB_SIZE]
            ),
            torch.bfloat16,
        ),
    }

    trainer = trainer_cls(rank, args, tokenizer, distill_metadata)
    trainer.train(train_dataloader)
    # trainer.save_pretrained("ckpts/fast-trained")


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU distributed two-stage forward example")

    parser.add_argument(
        "--student_devices", type=list, default=[0, 1, 2, 3], help="Devices for student model"
    )
    parser.add_argument(
        "--teacher_devices", type=list, default=[4, 5], help="Devices for teacher model"
    )
    parser.add_argument(
        "--teacher_parallel",
        type=str,
        choices=["tp", "mp"],
        default="tp",
        help="Parallel type for teacher model. TP and MP supported.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/magpie_llama3.2_1b_generated/data.cleaned.jsonl",
        help="Path to the training data.",
    )
    parser.add_argument(
        "--lazy_preprocess", type=bool, default=True, help="Whether to use lazy preprocessing."
    )
    parser.add_argument(
        "--out_path", type=str, default="ckpts/fast-trained", help="Path to save the model."
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--master_port", type=str, default="12357", help="Master port.")

    args = parser.parse_args()
    args.eagle_config = EAGLE3_DEFAULT_CFG["config"]
    # TODO: add sanity check for args

    def set_ranks(student_devices, teacher_devices, teacher_parallel):
        # TODO(hg): add "no-parallel" option, fallback when only one teacher device is provided.
        # TODO(hg): add "FSDP" option.
        if teacher_parallel == "tp":
            world_size = len(teacher_devices) + len(student_devices)
            student_ranks = list(range(len(student_devices)))
            teacher_ranks = list(
                range(len(student_devices), len(student_devices) + len(teacher_devices))
            )
        elif teacher_parallel == "mp":
            raise NotImplementedError("MP parallel type not supported.")
            # world_size = 2
            # student_rank = 0
            # teacher_ranks = [1]
        else:
            raise NotImplementedError(f"Parallel type {teacher_parallel} not supported.")
        return world_size, student_ranks, teacher_ranks

    args.world_size, args.student_ranks, args.teacher_ranks = set_ranks(
        args.student_devices, args.teacher_devices, args.teacher_parallel
    )

    # Launch multiple processes
    mp.spawn(
        train,
        args=(args,),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
