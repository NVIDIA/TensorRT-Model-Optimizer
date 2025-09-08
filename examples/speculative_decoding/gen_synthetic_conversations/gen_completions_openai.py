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

"""Generate synthetic conversational data by querying OpenAI-compatible endpoints."""

import argparse
import asyncio
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import transformers
from openai import AsyncOpenAI
from tqdm import tqdm as tqdm_sync
from utils import AsyncPrioritySemaphore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""Generate synthetic conversational data
        by sending prompts to one or more OpenAI-compatible HTTP endpoints"""
    )

    ## Model & Generation Parameters ##
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the served model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=None,
        help="""Maximum number of tokens to allow in the prompt.
        If provided, the model's tokenizer is used to count tokens and
        skip prompts that exceed this limit.""",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to allow in the completion.",
    )

    ## Client Parameters ##
    parser.add_argument(
        "--base-urls",
        type=str,
        default="http://localhost:8000/v1",
        help="""Comma-separated list of base URLs for OpenAI-compatible endpoints
        (e.g., http://localhost:8000/v1,http://localhost:8001/v1).
        If more than one URL is provided,
        the script will round-robin requests across them.""",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=512,
        help="Maximum concurrent requests allowed per client.",
    )
    parser.add_argument(
        "--openai-api-key",
        default="EMPTY",
        help="""Access key required by the OpenAI Python client
        (not required for local serving engines like vLLM).""",
    )

    ## I/O Parameters ##
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("input_conversations/train.jsonl"),
        help="""Path to the input `jsonl` file containing input conversations.
        Default is 'input_conversations/train.jsonl'.
        Alternatively, you can specify a directory and all `jsonl` files will be processed.
        Only the first turn will be used as a prompt.""",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("synthetic_conversations/"),
        help="""Directory to save the output `jsonl` files.
        Default is 'synthetic_conversations/'.
        This directory will be created if it does not exist, and will also contain checkpoints
        for partial results and resumed runs in the event of a crash.
        Output files will have the same names as the inputs.
        """,
    )
    parser.add_argument(
        "--num-checkpoints",
        type=int,
        default=10,
        help="""Number of checkpoints to save during processing.
        Each checkpoint will contain a portion of the processed conversations.
        Default number of checkpoints is 10.""",
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="""If set, checkpoint files will be retained after successful processing. Otherwise,
        they will be deleted to save space as long as fewer than 1%% of requests failed.""",
    )
    parser.add_argument(
        "--debug-max-num-conversations-per-file",
        type=int,
        default=None,
        help="""For debugging purposes,
        limit the number of conversations processed per file.
        If set, only this many conversations will be processed from each file.
        Default is None, meaning no limit.""",
    )

    return parser.parse_args()


## Checkpointing Logic ##


@dataclass
class CheckpointRecord:
    """One output entry for a processed conversation."""

    conversation_id: str
    conversations: list[dict]


@dataclass
class CheckpointState:
    """State for managing checkpointing during conversation processing."""

    # Index of this checkpoint relative to total number of checkpoints
    index: int

    # Number of requests expected for this checkpoint
    num_requests_for_checkpoint: int

    # Path to save the checkpoint file
    checkpoint_output_file: Path

    # List of processed conversation outputs
    outputs: list[CheckpointRecord | None] = field(default_factory=list)


def load_existing_conversation_ids(checkpoint_file: Path) -> list[str]:
    existing_ids = []
    if not checkpoint_file.exists():
        return existing_ids
    with checkpoint_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if not entry or "conversation_id" not in entry:
                continue
            existing_ids.append(entry["conversation_id"])

    return existing_ids


def write_checkpoint(
    checkpoint_state: CheckpointState,
) -> None:
    output_entries = []
    for record in checkpoint_state.outputs:
        if record is None:
            continue
        output_entries.append(
            {"conversation_id": record.conversation_id, "conversations": record.conversations}
        )

    if not output_entries:
        print(
            "No valid responses to write. "
            f"Skipping checkpoint at {checkpoint_state.checkpoint_output_file!s}"
        )
        return
    with checkpoint_state.checkpoint_output_file.open("a", encoding="utf-8") as f:
        for entry in output_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def add_result_to_checkpoint(
    conversation_obj: CheckpointRecord,
    checkpoint_state: CheckpointState,
) -> None:
    checkpoint_state.outputs.append(conversation_obj)
    if len(checkpoint_state.outputs) == checkpoint_state.num_requests_for_checkpoint:
        write_checkpoint(checkpoint_state)


@dataclass
class InputPromptData:
    """Data structure for an input prompt to be sent to the model."""

    prompt_conversation: list[dict]
    conversation_id: str
    source_filename: str


async def query_and_process_response(
    client: AsyncOpenAI,
    semaphore: AsyncPrioritySemaphore,
    prompt_data: InputPromptData,
    out_checkpoint_state: CheckpointState,
    args: argparse.Namespace,
    pbar: tqdm_sync,
) -> int:
    """Send a completion request while respecting per-client concurrency."""
    await semaphore.acquire(priority=out_checkpoint_state.index)
    try:
        response = await client.chat.completions.create(
            model=args.model,
            messages=prompt_data.prompt_conversation,
            temperature=args.temperature,
            max_tokens=args.max_completion_tokens,
        )
    except Exception:
        response = None
    finally:
        await semaphore.release()

    if response is None or not response.choices or not response.choices[0].message.content.strip():
        # Request failed or returned empty response
        out_checkpoint_state.outputs.append(None)
        pbar.update(1)
        return 0

    response_text = response.choices[0].message.content.strip()
    updated_conversation = [
        *prompt_data.prompt_conversation,
        {"role": "assistant", "content": response_text},
    ]
    output_obj = CheckpointRecord(
        conversation_id=prompt_data.conversation_id,
        conversations=updated_conversation,
    )
    add_result_to_checkpoint(output_obj, out_checkpoint_state)
    pbar.update(1)

    return 1  # Indicate that a valid response was processed


async def main(args: argparse.Namespace) -> None:
    client_base_paths = args.base_urls.split(",")
    if not client_base_paths:
        msg = "No base URLs provided for OpenAI clients."
        raise ValueError(msg)

    # Initialize clients and synchronization primitives (for concurrency control)
    clients: list[AsyncOpenAI] = []
    semaphores: list[asyncio.Semaphore] = []
    for i in range(len(client_base_paths)):
        base_url = client_base_paths[i]
        custom_http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=2048, max_keepalive_connections=1024),
            timeout=httpx.Timeout(timeout=None),
        )
        clients.append(
            AsyncOpenAI(
                base_url=base_url,
                api_key=args.openai_api_key,
                http_client=custom_http_client,
            )
        )
        semaphores.append(AsyncPrioritySemaphore(args.max_concurrent))

    # Load tokenizer to count prompt tokens
    if args.max_prompt_tokens is not None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)

    # Load input data
    input_files: list[Path] = []
    if args.input_path.is_dir():
        input_files.extend(args.input_path.glob("*.jsonl"))
    else:
        if args.input_path.suffix != ".jsonl" or not args.input_path.exists():
            msg = f"Input path {args.input_path} is not a valid .jsonl file or directory."
            raise ValueError(msg)
        input_files.append(args.input_path)

    all_prompts: list[InputPromptData] = []
    num_removed_too_long = 0
    num_first_turn_not_user = 0
    for input_file in input_files:
        num_requests_per_split = 0
        with input_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if not entry or "conversations" not in entry or not entry["conversations"]:
                    continue
                prompt = entry["conversations"][0]
                if not prompt or "role" not in prompt or prompt["role"] != "user":
                    num_first_turn_not_user += 1
                    continue
                conversation_id = entry["conversation_id"]
                if not conversation_id:
                    msg = f"Entry in file {input_file} is missing conversation_id."
                    raise ValueError(msg)
                msgs = [prompt]
                if args.max_prompt_tokens is not None:
                    assert tokenizer is not None, (
                        "Tokenizer should be initialized if max_prompt_tokens is set."
                    )
                    if len(tokenizer.apply_chat_template(msgs)) > args.max_prompt_tokens:
                        num_removed_too_long += 1
                        continue
                all_prompts.append(
                    InputPromptData(
                        prompt_conversation=msgs,
                        conversation_id=conversation_id,
                        source_filename=input_file.name,
                    )
                )
                num_requests_per_split += 1
                if (
                    args.debug_max_num_conversations_per_file
                    and num_requests_per_split >= args.debug_max_num_conversations_per_file
                ):
                    break

    # Shuffle requests
    random.seed(42)
    random.shuffle(all_prompts)

    print(f"Loaded {len(all_prompts)} total prompts from {len(input_files)} input files.")
    if num_removed_too_long > 0:
        print(f"Skipped {num_removed_too_long} prompts exceeding max token limit.")
    if num_first_turn_not_user > 0:
        print(f"Skipped {num_first_turn_not_user} prompts whose first turn was not 'user'.")

    # Initialize checkpointing state:
    # First, initialize the checkpoints and load the ids of requests already saved
    checkpoint_states = []
    num_requests_per_checkpoint = len(all_prompts) // args.num_checkpoints
    if args.num_checkpoints <= 0:
        msg = "Number of checkpoints must be greater than 0"
        raise ValueError(msg)

    all_processed_ids = set()
    for i in range(args.num_checkpoints):
        checkpoint_save_path = args.output_dir / f"checkpoints/checkpoint_{i}.jsonl"
        checkpoint_save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_states.append(
            CheckpointState(
                index=i,
                num_requests_for_checkpoint=0,
                checkpoint_output_file=checkpoint_save_path,
            )
        )
        all_processed_ids.update(load_existing_conversation_ids(checkpoint_save_path))

    print(f"Found {len(all_processed_ids)} already-processed conversations from checkpoints.")
    new_prompts = [p for p in all_prompts if p.conversation_id not in all_processed_ids]
    print(f"{len(new_prompts)} conversations remain to be processed after filtering.")

    for i in range(len(new_prompts)):
        checkpoint_idx_for_request = min(i // num_requests_per_checkpoint, args.num_checkpoints - 1)
        checkpoint_states[checkpoint_idx_for_request].num_requests_for_checkpoint += 1

    # Now, process the new requests
    pbar = tqdm_sync(
        total=len(new_prompts),
        desc="Processing prompts",
        dynamic_ncols=True,
        smoothing=0.05,
    )
    promises = []
    for i, req_obj in enumerate(new_prompts):
        checkpoint_idx_for_request = min(i // num_requests_per_checkpoint, args.num_checkpoints - 1)
        promises.append(
            query_and_process_response(
                clients[i % len(clients)],
                semaphores[i % len(semaphores)],
                req_obj,
                checkpoint_states[checkpoint_idx_for_request],
                args,
                pbar,
            )
        )

    await asyncio.gather(*promises)
    pbar.close()

    print("All requests processed. Collating checkpoints...")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    # Load all checkpoints and collate conversations by ID
    all_conversation_dict = {}
    for checkpoint_state in checkpoint_states:
        assert len(checkpoint_state.outputs) == checkpoint_state.num_requests_for_checkpoint, (
            "Mismatch in number of processed requests for checkpoint."
        )
        if checkpoint_state.checkpoint_output_file.exists():
            with checkpoint_state.checkpoint_output_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    if not entry or "conversation_id" not in entry:
                        continue
                    all_conversation_dict[entry["conversation_id"]] = entry["conversations"]

    # Group loaded conversations by their original input file
    results_by_file = {}
    num_skipped = 0
    num_successes = 0
    for prompt in all_prompts:
        if prompt.conversation_id not in all_conversation_dict:
            num_skipped += 1
        else:
            num_successes += 1
            if prompt.source_filename not in results_by_file:
                results_by_file[prompt.source_filename] = []
            results_by_file[prompt.source_filename].append(
                {
                    "conversation_id": prompt.conversation_id,
                    "conversations": all_conversation_dict[prompt.conversation_id],
                }
            )

    # Save all results to their respective output files
    for filename, entries in results_by_file.items():
        output_file_path = output_dir / filename
        with output_file_path.open("w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved {num_successes} conversations to {output_dir!s}")
    print(f"Failed to process {num_skipped} conversations.")

    # Remove checkpoint files if the failure rate was low enough
    if not args.keep_checkpoints and (num_skipped / max(1, len(new_prompts))) < 0.01:
        print("Removing checkpoint files...")
        for checkpoint_state in checkpoint_states:
            if checkpoint_state.checkpoint_output_file.exists():
                checkpoint_state.checkpoint_output_file.unlink()
        checkpoint_dir = args.output_dir / "checkpoints"
        if checkpoint_dir.exists() and checkpoint_dir.is_dir():
            checkpoint_dir.rmdir()
        print("Checkpoint files removed.")
    elif not args.keep_checkpoints:
        print(
            "Not removing checkpoint files due to high failure rate. "
            "You can manually delete them if desired, "
            "or rerun the script to retry the failed requests."
        )


if __name__ == "__main__":
    cli_args = parse_args()
    asyncio.run(main(cli_args))
