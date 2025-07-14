# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path

import nemo_run as run
from megatron.core.dist_checkpointing.validation import StrictHandling
from nemo.collections import llm
from nemo.collections.llm.api import export_ckpt
from nemo.collections.llm.gpt.data import ChatDataModule, MockDataModule
from nemo.collections.llm.modelopt.recipes.distillation_recipe import distillation_recipe
from nemo.collections.llm.modelopt.recipes.prune_recipe import prune_recipe
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer


def get_args():
    parser = argparse.ArgumentParser(
        description="NeMo 2.0 Pruning + Distillation flow. Currently supports models that fit on 1 node and 8 GPUs."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Experiment name",
        default="prune_distill_flow",
    )
    parser.add_argument(
        "--model_hf",
        type=str,
        help="Hugging Face model name or path",
        default="meta-llama/Meta-Llama-3.1-8B",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default="llama31_8b",
        help=(
            "Choose NeMo 2.0 recipe. Recipes are named in the format of "
            "<model_name>_<model_size>(_<long_sequence_length> or other special settings)"
        ),
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the finetuning chat dataset. Can be either ShareGPT or HuggingFace/OpenAI chat format",
    )
    parser.add_argument(
        "--tokenizer_hf",
        type=str,
        help="Hugging Face tokenizer name or path",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        help="Path to the custom chat template to replace the HF tokenizer default chat template.",
        required=False,
    )
    parser.add_argument(
        "--prune_target_hidden_size",
        type=int,
        default=3072,
        help="Target hidden size for pruning (also sets ffn_hidden_size to 3 * target_hidden_size)",
    )
    parser.add_argument(
        "--mock_run",
        action="store_true",
        help="Run in mock mode",
    )
    return parser.parse_args()


def get_finetune_recipe(recipe_name: str):
    if not hasattr(getattr(llm, recipe_name), "finetune_recipe"):
        raise ValueError(f"Recipe {recipe_name} does not have a Fine-Tuning recipe")
    return getattr(llm, recipe_name).finetune_recipe(peft_scheme=None)


def get_most_recent_subdir(directory: Path):
    # Get all subdirectories
    subdirs = [d for d in directory.iterdir() if d.is_dir()]
    if not subdirs:
        raise ValueError(f"No subdirectories found in {directory}")

    # Sort by modification time (most recent first)
    most_recent = max(subdirs, key=lambda x: x.stat().st_mtime)

    return most_recent


def get_most_recent_ckpt(directory: str):
    """
    Find the most recent checkpoint subdirectory in a given directory.

    Args:
        directory (str): Path to the directory to search in.

    Returns:
        str: Path to the most recent subdirectory.
    """
    exp_dir = Path(directory) / "default"
    assert exp_dir.exists(), f"Experiment directory {exp_dir} does not exist"

    checkpoint_dir = exp_dir / "checkpoints"
    if checkpoint_dir.exists():
        most_recent = get_most_recent_subdir(checkpoint_dir)
    else:
        most_recent = get_most_recent_subdir(exp_dir)
        checkpoint_dir = most_recent / "checkpoints"
        assert checkpoint_dir.exists(), f"Checkpoint directory {checkpoint_dir} does not exist"
        most_recent = get_most_recent_subdir(checkpoint_dir)

    return str(most_recent)


def export_most_recent_ckpt(directory: str, output_path: str):
    most_recent_ckpt = get_most_recent_ckpt(directory)
    export_ckpt(most_recent_ckpt, "hf", output_path=output_path, overwrite=True)


def _read_chat_template(template_path: str):
    with open(template_path) as f:
        return f.read().strip()


if __name__ == "__main__":
    args = get_args()

    # # # # # CONFIGURABLE PARAMETERS # # # # #
    GPUS = 8
    SEQUENCE_LENGTH = 8192
    MBS = 2
    VAL_BATCHES = 32
    if args.mock_run:
        GBS = 8
        FINETUNE_STEPS = 20
        DISTILL_STEPS = 20
        VAL_INTERVAL = 10
        PRUNE_SAMPLES = 3
        assert args.data_path is None, "Argument --data_path not used in mock mode."
    else:
        GBS = 2048
        FINETUNE_STEPS = 2500
        DISTILL_STEPS = 7500
        VAL_INTERVAL = 500
        PRUNE_SAMPLES = 1024
        args.data_path = args.data_path if args.data_path is not None else "lima_processed"
    # # # # # # # # # # # # # # # # # # # # # #

    # Common items
    model_name = args.recipe
    model_module = getattr(llm, model_name)
    if args.mock_run:
        data = run.Config(
            MockDataModule,
            global_batch_size=GBS,
            micro_batch_size=MBS,
            seq_length=SEQUENCE_LENGTH,
        )
    else:
        tokenizer = run.Config(
            get_nmt_tokenizer,
            library="huggingface",
            model_name=args.tokenizer_hf,
            chat_template=_read_chat_template(args.chat_template) if args.chat_template else None,
        )
        data = run.Config(
            ChatDataModule,
            dataset_root=args.data_path,
            seq_length=SEQUENCE_LENGTH,
            tokenizer=tokenizer,
            global_batch_size=GBS,
            micro_batch_size=MBS,
            use_hf_tokenizer_chat_template=True,
        )

    # 1. Process data and Import and save initial NeMo checkpoint
    lima_data = run.Script("process_lima.py", entrypoint="python")

    initial_model_out = f"{args.experiment_name}/{model_name}_initial"
    import_model = run.Partial(
        llm.import_ckpt,
        model=model_module.model(),
        source=f"hf://{args.model_hf}",
        output_path=initial_model_out,
    )

    # 2. Finetune
    finetune = get_finetune_recipe(args.recipe)
    finetune.tokenizer = "data"
    finetune.data = data
    finetune.resume.restore_config.path = initial_model_out
    finetune.log.log_dir = f"{args.experiment_name}/finetune_log_dir"
    finetune.trainer.strategy.tensor_model_parallel_size = GPUS
    finetune.trainer.max_steps = FINETUNE_STEPS
    finetune.trainer.val_check_interval = VAL_INTERVAL
    finetune.trainer.limit_val_batches = VAL_BATCHES

    # 3. Prune to obtain student
    prune_data = data.clone()
    prune_data.global_batch_size = prune_data.micro_batch_size
    finetuned_teacher_path = run.Config(get_most_recent_ckpt, finetune.log.log_dir)
    pruned_model_out = f"{args.experiment_name}/{model_name}_pruned"
    prune = prune_recipe(
        nemo_checkpoint=finetuned_teacher_path,
        save_path=pruned_model_out,
    )
    prune.pruning_config.target_hidden_size = args.prune_target_hidden_size
    prune.pruning_config.target_ffn_hidden_size = args.prune_target_hidden_size * 3
    prune.devices = GPUS
    prune.pp_size = GPUS
    prune.data = prune_data
    prune.num_train_samples = PRUNE_SAMPLES
    prune.legacy_ckpt = True

    # 4. Distill
    distill = distillation_recipe(
        teacher_model_path=finetuned_teacher_path,
        student_model_path=pruned_model_out,
    )
    distill.data = data
    distill.tokenizer = "data"
    distill.log.log_dir = f"{args.experiment_name}/distill_log_dir"
    distill.trainer.max_steps = DISTILL_STEPS
    distill.trainer.val_check_interval = VAL_INTERVAL
    distill.trainer.limit_val_batches = VAL_BATCHES
    distill.trainer.strategy.ckpt_load_strictness = StrictHandling.LOG_ALL

    # 5. Export
    export_path = f"{args.experiment_name}/{model_name}_final"
    export_model = run.Partial(
        export_most_recent_ckpt,
        directory=f"{args.experiment_name}/distill_log_dir",
        output_path=export_path,
    )

    # Run all
    executor_single = run.LocalExecutor()
    executor_multi = run.LocalExecutor(launcher="torchrun", ntasks_per_node=GPUS)
    with run.Experiment(args.experiment_name, log_level="INFO") as exp:
        if args.data_path == "lima_processed":
            s0 = exp.add(lima_data, tail_logs=True, name="lima_data", executor=run.LocalExecutor())
        s1 = exp.add(import_model, executor=executor_single, tail_logs=True, name="import")
        s2 = exp.add(
            finetune, executor=executor_multi, tail_logs=True, name="finetune", dependencies=[s1]
        )
        s3 = exp.add(
            prune, executor=executor_multi, tail_logs=True, name="prune", dependencies=[s2]
        )
        s4 = exp.add(
            distill, executor=executor_multi, tail_logs=True, name="distill", dependencies=[s3]
        )
        s5 = exp.add(
            export_model, executor=executor_single, tail_logs=True, name="export", dependencies=[s4]
        )
        exp.run()
