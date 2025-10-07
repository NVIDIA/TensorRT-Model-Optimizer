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
from distill_trainer import EagleTPTrainer
from eagle_utils import DataCollatorWithPadding, make_eagle_supervised_data_module
from transformers import AutoTokenizer

# Hyperparameters for profiling
torch.manual_seed(0)


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
    args.teacher_pgroup = dist.new_group(ranks=args.teacher_ranks)
    args.student_pgroup = dist.new_group(ranks=args.student_ranks)


def train(rank, args):
    _setup_distributed(rank, args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, model_max_length=args.training_seq_len
    )
    data_module = make_eagle_supervised_data_module(tokenizer, args, use_offline_training=False)

    train_dataloader = torch.utils.data.DataLoader(
        data_module["train_dataset"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=DataCollatorWithPadding(max_length=args.training_seq_len),
        drop_last=True,
    )

    trainer = EagleTPTrainer(rank, args, tokenizer, train_dataloader)
    trainer.train()
    trainer.save_pretrained(args.out_path)


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU distributed two-stage forward example")
    parser.add_argument("--model_path", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--student_devices", type=list, default=[0, 1, 2, 3])
    parser.add_argument("--teacher_devices", type=list, default=[4, 5, 6, 7])
    parser.add_argument(
        "--data_path", type=str, default="data/magpie_llama3.2_1b_generated/data.cleaned.jsonl"
    )
    parser.add_argument("--training_seq_len", type=str, default=1024)
    parser.add_argument("--eagle_config_path", type=str, default="eagle_config.json")
    parser.add_argument(
        "--lazy_preprocess", type=bool, default=True, help="Whether to use lazy preprocessing."
    )
    parser.add_argument("--out_path", type=str, default="ckpts/fast-trained")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Total batch size across all parallel ranks."
    )
    parser.add_argument("--master_port", type=str, default="12357")

    args = parser.parse_args()
    # TODO: add sanity check for args

    def set_ranks(args):
        # TODO(hg): This is for TP-DDP setting only. Add "no-parallel", "MP", "FSDP".
        args.world_size = len(args.teacher_devices) + len(args.student_devices)
        args.student_ranks = list(range(len(args.student_devices)))
        args.teacher_ranks = list(
            range(len(args.student_devices), len(args.student_devices) + len(args.teacher_devices))
        )

    set_ranks(args)
    # Launch multiple processes
    mp.spawn(
        train,
        args=(args,),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
