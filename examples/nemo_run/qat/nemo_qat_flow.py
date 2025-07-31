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
from pathlib import Path

import nemo_run as run
from nemo.collections import llm
from nemo.collections.llm.api import export_ckpt
from nemo.collections.llm.gpt.data.chat import ChatDataModule
from nemo.collections.llm.modelopt.quantization.quant_cfg_choices import get_quant_cfg_choices
from nemo.collections.llm.modelopt.recipes.distillation_recipe import distillation_recipe
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.utils import logging


def get_parser():
    parser = argparse.ArgumentParser(
        description="NeMo2.0 QAT/QAD simplified flow. Currently supports running model locally on 1 node with 8 GPUs."
    )
    quant_cfg_choices_list = ["no_quant", *get_quant_cfg_choices()]

    parser.add_argument(
        "--model-name",
        type=str,
        help="Name of the HF model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--finetune-recipe",
        type=str,
        default="llama31_8b",
        help=(
            "Choose NeMo 2.0 recipe. Recipes are named in the format of "
            "<model_name>_<model_size>(_<long_sequence_length> or other special settings)"
        ),
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the finetuning chat dataset. Can be either ShareGPT or HuggingFace/OpenAI chat format",
    )
    parser.add_argument(
        "--distill",
        action="store_true",
        help="Whether to do quantization aware distillation (QAD)",
    )
    parser.add_argument(
        "--hf-tokenizer",
        type=str,
        help="Name of HF model to use for tokenizer.",
        required=False,
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        help="Path to the custom chat template to replace the HF tokenizer default chat template.",
        required=False,
    )
    parser.add_argument(
        "-algo",
        "--algorithm",
        type=str,
        default="fp8",
        choices=quant_cfg_choices_list,
        help="TensorRT-Model-Optimizer quantization algorithm",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Run on slurm using run.SlurmExecutor",
        default=False,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment name",
        default="qat_flow_ckpts",
    )
    parser.add_argument(
        "--ptq_gpus",
        type=int,
        help="Number of GPUs for quantization. Some models require a different number of GPUs for PTQ vs training.",
        default=8,
    )
    parser.add_argument(
        "--kv-cache-qformat",
        type=str,
        default="fp8",
        choices=["fp8", "nvfp4"],
        help="KV-cache quantization format",
    )
    parser.add_argument(
        "--enable_kv_cache", help="Enables KV-cache quantization", action="store_true"
    )
    parser.add_argument("--disable_kv_cache", dest="enable_kv_cache", action="store_false")
    parser.set_defaults(enable_kv_cache=None)
    return parser


def get_finetune_recipe(recipe):
    assert hasattr(llm, recipe), (
        f"Recipe named {recipe} not found. General format is <model_name>_<model_size>(_<long_sequence_length> "
        "or other special settings)"
    )
    finetune_recipe = getattr(llm, recipe).finetune_recipe
    return finetune_recipe(peft_scheme=None)  # TODO add dir


def get_most_recent_subdir(directory: str):
    """
    Find the most recent subdirectory in a given directory.

    Args:
        directory (str): Path to the directory to search in

    Returns:
        str: Path to the most recent subdirectory, or None if no subdirectories exist
    """
    dir_path = Path(directory)
    # Get all subdirectories
    subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
    if not subdirs:
        return None

    # Sort by modification time (most recent first)
    most_recent = max(subdirs, key=lambda x: x.stat().st_mtime)
    return str(most_recent)


def export_most_recent_ckpt(exp_dir: str, output_path: str):
    """
    Args:
        exp_dir: experiment directory
        output_path: path to write exported model
    """
    most_recent_exp = get_most_recent_subdir(f"{exp_dir}/default/")
    if "checkpoints" in most_recent_exp:
        most_recent_ckpt = most_recent_exp
    else:
        most_recent_ckpt = get_most_recent_subdir(f"{most_recent_exp}/checkpoints/")
    logging.info(f"Exporting checkpoint from {most_recent_ckpt}")
    export_ckpt(most_recent_ckpt, "hf", output_path)


def _read_chat_template(template_path: str):
    with open(template_path) as f:
        return f.read().strip()


if __name__ == "__main__":
    args = get_parser().parse_args()
    if not args.distill and not args.finetune_recipe:
        raise ValueError("If distillation is not used, --finetune-recipe must be specified")
    model_name = args.finetune_recipe
    model_module = getattr(llm, model_name)
    if not model_name:
        model_name = os.path.basename(args.model_name)

    # 1. Process data
    lima_data = run.Script("process_lima.py", entrypoint="python")

    # 2. Import Model
    nemo_ckpt_path = f"{model_name}-nemo"
    import_model = run.Partial(
        llm.import_ckpt,
        model=model_module.model(),
        source=f"hf://{args.model_name}",
        output_path=nemo_ckpt_path,
    )
    # 3. PTQ
    ptq_model_out = f"{model_name}-{args.algorithm}"

    ptq = run.Script(
        "/opt/NeMo/scripts/llm/ptq.py",
        args=[
            "-nc",
            nemo_ckpt_path,
            "-out",
            ptq_model_out,
            "--export_format",
            "nemo",
            "--algorithm",
            args.algorithm,
            "--kv_cache_qformat",
            args.kv_cache_qformat,
            "--enable_kv_cache" if args.enable_kv_cache else "--disable_kv_cache",
            "-ctp",
            f"{args.ptq_gpus}",
        ],
        entrypoint="python",
    )

    # 4. Train
    if not args.hf_tokenizer:
        tokenizer_path = os.path.join(nemo_ckpt_path, "context/nemo_tokenizer")
        tokenizer = run.Config(
            get_nmt_tokenizer,
            library="huggingface",
            model_name=tokenizer_path,
            chat_template=_read_chat_template(args.chat_template) if args.chat_template else None,
        )
    else:
        tokenizer = run.Config(
            get_nmt_tokenizer,
            library="huggingface",
            model_name=args.hf_tokenizer,
            chat_template=_read_chat_template(args.chat_template) if args.chat_template else None,
        )

    data_path = args.data_path if args.data_path is not None else "lima_processed"
    data = run.Config(
        ChatDataModule,
        dataset_root=data_path,
        seq_length=4096,
        tokenizer=tokenizer,
        global_batch_size=64,
        micro_batch_size=1,
        use_hf_tokenizer_chat_template=True,
    )
    if args.distill:
        train = distillation_recipe(ptq_model_out, nemo_ckpt_path)
    else:
        train = get_finetune_recipe(args.finetune_recipe)
        train.resume.restore_config.path = ptq_model_out
    train.tokenizer = "data"
    train.data = data
    train.log.log_dir = args.experiment
    train.trainer.val_check_interval = 200
    train.trainer.max_steps = 200

    # 5. Export
    export = run.Partial(
        export_most_recent_ckpt, exp_dir=train.log.log_dir, output_path=f"{model_name}_hf"
    )

    with run.Experiment(args.experiment, log_level="INFO") as exp:
        ptq_executor = run.LocalExecutor(ntasks_per_node=args.ptq_gpus, launcher="torchrun")
        if not args.data_path:
            s0 = exp.add(lima_data, tail_logs=True, name="lima_data", executor=run.LocalExecutor())
        s1 = exp.add(
            import_model, tail_logs=True, name="import_model", executor=run.LocalExecutor()
        )
        s2 = exp.add(ptq, tail_logs=True, name="ptq", executor=ptq_executor, dependencies=[s1])
        train_executor = run.LocalExecutor(ntasks_per_node=8, launcher="torchrun")
        s3 = exp.add(
            train, tail_logs=True, name="train", executor=train_executor, dependencies=[s2]
        )
        s4 = exp.add(
            export,
            tail_logs=True,
            name="export_hf",
            executor=run.LocalExecutor(),
            dependencies=[s3],
        )
        exp.run(detach=False)
