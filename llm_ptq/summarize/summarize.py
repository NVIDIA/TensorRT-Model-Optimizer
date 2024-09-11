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

import argparse
import ast
import os
from pathlib import Path

import evaluate
import numpy as np
import tensorrt_llm
import tensorrt_llm.profiler as profiler
import torch
from datasets import load_dataset
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.models.qwen.utils import make_context
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner
from tensorrt_llm.tools.ppl import ppl
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, GenerationConfig
from utils import (
    DEFAULT_HF_MODEL_DIRS,
    add_common_args,
    load_tokenizer,
    read_model_name,
    supports_inflight_batching,
)

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp


def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    test_hf = args.test_hf and runtime_rank == 0  # only run hf on rank 0
    test_trt_llm = args.test_trt_llm
    model_name, model_version = read_model_name(args.engine_dir)
    if args.hf_model_dir is None:
        logger.warning(
            "hf_model_dir is not specified. Try to infer from model_name, but this may be incorrect."
        )
        if model_name in DEFAULT_HF_MODEL_DIRS:
            args.hf_model_dir = DEFAULT_HF_MODEL_DIRS[model_name]
        else:
            args.hf_model_dir = None
    if args.tokenizer_dir is None:
        args.tokenizer_dir = args.hf_model_dir

    profiler.start("load tokenizer")
    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        vocab_file=args.vocab_file,
        model_name=model_name,
        model_version=model_version,
    )
    profiler.stop("load tokenizer")
    logger.info(f'Load tokenizer takes: {profiler.elapsed_time_in_sec("load tokenizer")} sec')

    if args.eval_task == "code_completion":
        dataset_name = "openai_humaneval"
        dataset_revision = None
        dataset_input_key = "prompt"
        dataset_output_key = "canonical_solution"
        dataset_split = "test"
    elif args.eval_task == "summarize":
        dataset_name = "ccdv/cnn_dailymail"
        dataset_revision = "3.0.0"
        dataset_input_key = "article"
        dataset_output_key = "highlights"
        dataset_split = "test"
    elif args.eval_task == "summarize_long":
        dataset_name = "tau/zero_scrolls"
        dataset_revision = "squality"
        dataset_input_key = "input"
        dataset_output_key = "output"
        dataset_split = "validation"  # only this split contains reference strings
    elif args.eval_task == "eval_context_ppl":
        dataset_name = "SlimPajama-6B"
        dataset_revision = None
        dataset_input_key = "text"
        dataset_output_key = "text"
        dataset_split = "test"
        args.output_len = 1  # Only want to compute the ppl of context
        args.eval_ppl = True
        logger.warning(
            f"Run task '{args.eval_task}', setting 'output_len' to 1, and enable 'eval_ppl'."
        )
    if args.dataset_dir is not None and isinstance(args.dataset_dir, str):
        args.dataset_dir = args.dataset_dir.rstrip("/")
        if args.dataset_dir.endswith(dataset_name):
            dataset_name = args.dataset_dir
        else:
            dataset_name = f"{args.dataset_dir}/{dataset_name}"
    dataset = load_dataset(
        dataset_name,
        dataset_revision,
        cache_dir=args.dataset_cache_dir,
        split=dataset_split,
        trust_remote_code=True,
    )

    max_batch_size = args.batch_size

    # runtime parameters
    top_k = args.top_k
    top_p = args.top_p
    output_len = args.output_len
    test_token_num = args.max_input_length
    max_attention_window_size = args.max_attention_window_size
    sink_token_length = args.sink_token_length

    if args.end_id:
        end_id = args.end_id

    stop_words_list = None
    if args.stop_words:
        stop_words_list = tensorrt_llm.runtime.decode_words_list(args.stop_words, tokenizer)

    bad_words_list = None
    if args.bad_words:
        bad_words_list = tensorrt_llm.runtime.decode_words_list(args.bad_words, tokenizer)

    # random_seed = 5
    temperature = args.temperature
    num_beams = args.num_beams
    length_penalty = args.length_penalty
    early_stopping = args.early_stopping
    repetition_penalty = args.repetition_penalty
    presence_penalty = args.presence_penalty
    frequency_penalty = args.frequency_penalty

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        if test_trt_llm:
            with (output_dir / "trtllm.out").open("w") as f:
                f.write(f"Engine path: {args.engine_dir}\n")
                f.write(f"Tokenizer path: {args.tokenizer_dir}\n")
        if test_hf:
            with (output_dir / "hf.out").open("w") as f:
                f.write(f"Model path: {args.hf_model_dir}\n")
                f.write(f"Tokenizer path: {args.tokenizer_dir}\n")

    # TODO: Add random_seed flag in gptj
    rouge_dir = args.rouge_dir if args.rouge_dir and os.path.exists(args.rouge_dir) else "rouge"
    metric_tensorrt_llm = [evaluate.load(rouge_dir) for _ in range(num_beams)]
    metric_hf = [evaluate.load(rouge_dir) for _ in range(num_beams)]
    for i in range(num_beams):
        metric_tensorrt_llm[i].seed = 0
        metric_hf[i].seed = 0
    ppls_trt_llm = [[] for _ in range(num_beams)]
    ppls_hf = [[] for _ in range(num_beams)]

    def _prepare_inputs(
        batch_input_texts, eval_task="summarize", add_special_tokens=True, min_input_length=0
    ):
        batch_size = len(batch_input_texts)
        append_str = " TL;DR: " if eval_task == "summarize" else ""
        batch_input_ids = []
        for i in range(batch_size):
            curr_text = batch_input_texts[i] + append_str
            curr_text = curr_text.strip().replace(" n't", "n't")

            # TODO: The below lines are used to be compatible with the original code; may need fix
            if model_name == "ChatGLMForCausalLM" and model_version in ["chatglm2", "chatglm3"]:
                input_ids = tokenizer.encode(curr_text, return_tensors="pt").squeeze(0)
                input_ids = input_ids[:test_token_num]
            elif model_name == "QWenForCausalLM" and model_version == "qwen":
                # use make_content to generate prompt
                system_prompt = (
                    "You are a useful assistant, "
                    "please directly output the corresponding summary according to the article entered by the user."
                )
                _, input_id_list = make_context(
                    tokenizer=tokenizer,
                    query=curr_text,
                    history=[],
                    system=system_prompt,
                    max_input_length=test_token_num,
                )
                input_ids = torch.tensor(input_id_list)
            else:
                if model_name == "QWenForCausalLM" and model_version == "qwen2":
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": curr_text},
                    ]
                    curr_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                input_ids = tokenizer.encode(
                    curr_text,
                    return_tensors="pt",
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    max_length=test_token_num,
                ).squeeze(0)

            if input_ids.numel() > min_input_length:
                batch_input_ids.append(input_ids)
        return batch_input_ids

    def eval_trt_llm(
        datapoint,
        eval_task="summarize",
        eval_ppl=False,
        add_special_tokens=True,
        min_input_length=0,
    ):
        batch_size = len(datapoint[dataset_input_key])
        batch_input_ids = _prepare_inputs(
            datapoint[dataset_input_key],
            eval_task=eval_task,
            add_special_tokens=add_special_tokens,
            min_input_length=min_input_length,
        )
        batch_size = len(batch_input_ids)
        if batch_size == 0:
            return [], [], [], {}
        input_lengths = [x.size(0) for x in batch_input_ids]

        with torch.no_grad():
            outputs = runner.generate(
                batch_input_ids,
                max_new_tokens=output_len,
                max_attention_window_size=max_attention_window_size,
                sink_token_length=sink_token_length,
                end_id=end_id,
                pad_id=pad_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_words_list=stop_words_list,
                bad_words_list=bad_words_list,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                lora_uids=args.lora_task_uids,
                output_sequence_lengths=True,
                return_dict=True,
                medusa_choices=args.medusa_choices,
            )
            torch.cuda.synchronize()

        # Extract a list of tensors of shape beam_width x output_ids.
        if runtime_rank == 0:
            output_ids = outputs["output_ids"]
            output_beams_list = [
                tokenizer.batch_decode(
                    output_ids[batch_idx, :, input_lengths[batch_idx] :], skip_special_tokens=True
                )
                for batch_idx in range(batch_size)
            ]
            output_ids_list = [
                output_ids[batch_idx, :, input_lengths[batch_idx] :]
                for batch_idx in range(batch_size)
            ]

            ppls = [[] for _ in range(batch_size)]
            seq_lengths_array = outputs["sequence_lengths"].cpu().tolist()
            lengths_info = {"input_lengths": input_lengths, "seq_lengths": seq_lengths_array}
            if eval_ppl:
                seq_lengths = outputs["sequence_lengths"]
                context_logits = outputs["context_logits"]
                # Remove the first generation logits which are same to last context logits
                generation_logits = outputs["generation_logits"][:, :, 1:]
                for batch_idx in range(batch_size):
                    # [batch, beam, step]
                    for beam_idx in range(num_beams):
                        curr_len = seq_lengths[batch_idx, beam_idx]
                        curr_ctx_len = input_lengths[batch_idx]
                        curr_gen_len = curr_len - curr_ctx_len

                        curr_ids = output_ids[batch_idx, beam_idx, 1:curr_len]
                        curr_logits = torch.cat(
                            [
                                context_logits[batch_idx],
                                generation_logits[batch_idx, beam_idx, : curr_gen_len - 1],
                            ],
                            dim=0,
                        )
                        curr_ppl = ppl(curr_logits, curr_ids)
                        logger.debug(
                            f"TensorRT-LLM PPL: {curr_ppl:.3f} | Generation length: {curr_gen_len}"
                        )
                        ppls[batch_idx].append(curr_ppl)

            return output_beams_list, output_ids_list, ppls, lengths_info
        return [], [], [], {}

    def eval_hf(
        datapoint,
        eval_task="summarize",
        eval_ppl=False,
        add_special_tokens=True,
        min_input_length=0,
    ):
        batch_size = len(datapoint[dataset_input_key])
        if batch_size > 1:
            logger.warning(
                "HF does not support batch_size > 1 to verify correctness due to padding. Current"
                f" batch size is {batch_size}"
            )
        batch_input_ids = _prepare_inputs(
            datapoint[dataset_input_key],
            eval_task=eval_task,
            add_special_tokens=add_special_tokens,
            min_input_length=min_input_length,
        )
        batch_size = len(batch_input_ids)
        if batch_size == 0:
            return [], [], [], [[] for _ in range(batch_size)]
        input_lengths = [x.size(0) for x in batch_input_ids]
        # Left padding for HF
        max_length = max(input_lengths)
        paddings = [
            torch.ones(max_length - length, dtype=torch.int32) * pad_id for length in input_lengths
        ]
        batch_input_ids = [torch.cat([pad, x]) for x, pad in zip(batch_input_ids, paddings)]
        batch_input_ids = torch.stack(batch_input_ids)
        batch_input_ids = batch_input_ids.cuda()

        # specialization for HF
        if early_stopping in [0, 1]:
            local_early_stopping = bool(early_stopping)
        else:
            local_early_stopping = "never"

        with torch.no_grad():
            outputs = model.generate(
                batch_input_ids,
                max_new_tokens=output_len,
                top_k=top_k,
                temperature=temperature,
                eos_token_id=end_id,
                pad_token_id=pad_id,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                length_penalty=length_penalty,
                early_stopping=local_early_stopping,
                output_scores=True,
                return_dict_in_generate=True,
            )
            if eval_ppl and batch_size == 1:
                # model.generate cannot return context logits?
                # Will cause additional latency
                context_outputs = model(batch_input_ids)

        output_ids = outputs["sequences"]
        tokens_list = output_ids[:, len(batch_input_ids[0]) :].tolist()
        output_ids = output_ids.reshape([batch_size, num_beams, -1])
        output_lines_list = [
            tokenizer.batch_decode(
                output_ids[:, i, len(batch_input_ids[0]) :], skip_special_tokens=True
            )
            for i in range(num_beams)
        ]

        ppls = [[] for _ in range(batch_size)]
        if eval_ppl and batch_size == 1:
            # Only for batch size of 1
            seq_lens = (output_ids != end_id).logical_and(output_ids != pad_id).sum(dim=-1)
            context_logits = context_outputs["logits"]
            # Remove the first generation logits which are same to last context logits
            generation_logits = outputs["scores"][1:]
            # When output_len is 1, generation_logits would be () and lead to error if we do torch.stack
            if len(generation_logits) == 0:
                generation_logits = torch.empty(
                    [context_logits.shape[0], 0, context_logits.shape[-1]],
                    device=context_logits.device,
                )
            else:
                generation_logits = torch.stack(generation_logits, dim=1)
            _, max_gen_len, voc_size = generation_logits.size()
            generation_logits = generation_logits.view(batch_size, num_beams, max_gen_len, voc_size)
            for batch_idx in range(batch_size):
                for beam_idx in range(num_beams):
                    curr_len = seq_lens[batch_idx, beam_idx]
                    curr_ctx_len = input_lengths[batch_idx]
                    curr_gen_len = curr_len - curr_ctx_len

                    curr_ids = output_ids[batch_idx, beam_idx, 1:curr_len]
                    curr_logits = torch.cat(
                        [
                            context_logits[batch_idx],
                            generation_logits[batch_idx, beam_idx, : curr_gen_len - 1],
                        ],
                        dim=0,
                    )
                    curr_ppl = ppl(curr_logits, curr_ids)
                    logger.debug(f"HF PPL: {curr_ppl:.3f} | Generation length: {curr_gen_len}")
                    ppls[batch_idx].append(curr_ppl)

        return output_lines_list, tokens_list, ppls

    if test_trt_llm:
        if not supports_inflight_batching(args.engine_dir):
            logger.warning(
                "The given engine does not support in-flight batching, fallback to python session"
            )
            args.use_py_session = True

        if not PYTHON_BINDINGS and not args.use_py_session:
            logger.warning(
                "Python bindings of C++ session is unavailable, fallback to Python session."
            )
            args.use_py_session = True
        runner_cls = ModelRunner if args.use_py_session else ModelRunnerCpp
        runner_kwargs = dict(
            engine_dir=args.engine_dir,
            rank=runtime_rank,
            debug_mode=args.debug_mode,
            gpu_weights_percent=args.gpu_weights_percent,
        )
        if args.medusa_choices is not None:
            args.medusa_choices = ast.literal_eval(args.medusa_choices)
            assert args.temperature == 1.0, "Medusa should use temperature == 1.0"
            assert args.num_beams == 1, "Medusa should use num_beams == 1"
            runner_kwargs.update(medusa_choices=args.medusa_choices)
        if not args.use_py_session:
            runner_kwargs.update(
                max_batch_size=max_batch_size,
                max_input_len=test_token_num,
                max_output_len=output_len,
                max_beam_width=num_beams,
                max_attention_window_size=max_attention_window_size,
                sink_token_length=sink_token_length,
                max_tokens_in_paged_kv_cache=args.max_tokens_in_paged_kv_cache,
                kv_cache_enable_block_reuse=args.kv_cache_enable_block_reuse,
                kv_cache_free_gpu_memory_fraction=args.kv_cache_free_gpu_memory_fraction,
                enable_chunked_context=args.enable_chunked_context,
            )
        runner = runner_cls.from_dir(**runner_kwargs)
        assert not (
            args.eval_ppl and not (runner.gather_context_logits and runner.gather_generation_logits)
        ), "PPL evaluation requires engine built with gather_all_token_logits enabled"

        datapoint = dataset[0:1]
        output, *_ = eval_trt_llm(
            datapoint,
            eval_task=args.eval_task,
            eval_ppl=args.eval_ppl,
            add_special_tokens=args.add_special_tokens,
            min_input_length=args.min_input_length,
        )
        if runtime_rank == 0 and args.eval_task != "eval_context_ppl":
            logger.info("---------------------------------------------------------")
            logger.info("TensorRT-LLM Generated : ")
            logger.info(f" Input : {datapoint[dataset_input_key]}")
            logger.info(f"\n Reference : {datapoint[dataset_output_key]}")
            logger.info(f"\n Output : {output}")
            logger.info("---------------------------------------------------------")

        ite_count = 0
        data_point_idx = 0
        total_output_token_count_trt_llm = 0  # only valid for runtime_rank == 0
        while (data_point_idx < len(dataset)) and (ite_count < args.max_ite):
            if runtime_rank == 0:
                logger.debug(f"run data_point {data_point_idx} ~ {data_point_idx + max_batch_size}")
            datapoint = dataset[data_point_idx : (data_point_idx + max_batch_size)]

            profiler.start("tensorrt_llm")
            output_tensorrt_llm, output_ids_trt_llm, curr_ppls_trt_llm, lengths_info = eval_trt_llm(
                datapoint,
                eval_task=args.eval_task,
                eval_ppl=args.eval_ppl,
                add_special_tokens=args.add_special_tokens,
                min_input_length=args.min_input_length,
            )
            if output_tensorrt_llm == []:
                data_point_idx += max_batch_size
                ite_count += 1
                continue
            profiler.stop("tensorrt_llm")
            if runtime_rank == 0:
                input_lengths = lengths_info["input_lengths"]
                seq_lengths = lengths_info["seq_lengths"]
                output_token_count_trt_llm = sum(
                    seq_lengths[bs][bm] - input_lengths[bs]
                    for bm in range(len(output_tensorrt_llm[0]))
                    for bs in range(len(output_tensorrt_llm))
                )
                total_output_token_count_trt_llm += output_token_count_trt_llm

                for batch_idx in range(len(output_tensorrt_llm)):
                    for beam_idx in range(num_beams):
                        metric_tensorrt_llm[beam_idx].add_batch(
                            predictions=[output_tensorrt_llm[batch_idx][beam_idx]],
                            references=[datapoint[dataset_output_key][batch_idx]],
                        )
                        if args.eval_ppl:
                            ppls_trt_llm[beam_idx].append(curr_ppls_trt_llm[batch_idx][beam_idx])
                if output_dir is not None:
                    for i in range(len(output_tensorrt_llm[0])):
                        for beam_idx in range(num_beams):
                            with (output_dir / "trtllm.out").open("a") as f:
                                f.write(
                                    f"[{data_point_idx + i}] [Beam {beam_idx}] {output_tensorrt_llm[beam_idx][i]}\n"
                                )

                logger.debug("-" * 100)
                logger.debug(f"Input : {datapoint[dataset_input_key]}")
                logger.debug(f"TensorRT-LLM Output: {output_tensorrt_llm}")
                logger.debug(f"Reference : {datapoint[dataset_output_key]}")

            data_point_idx += max_batch_size
            ite_count += 1
        del runner

    if test_hf:
        profiler.start("load HF model")
        dtype_alias_mapping = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
        args.hf_data_type = dtype_alias_mapping.get(args.hf_data_type, args.hf_data_type)
        if model_name == "ChatGLMForCausalLM" and model_version == "glm":
            auto_model_cls = AutoModelForSeq2SeqLM
        elif model_name == "ChatGLMForCausalLM" and model_version == "chatglm":
            auto_model_cls = AutoModel
        else:
            auto_model_cls = AutoModelForCausalLM
        model = auto_model_cls.from_pretrained(
            args.hf_model_dir,
            trust_remote_code=True,
            torch_dtype=str_dtype_to_torch(args.hf_data_type),
            device_map="auto" if args.hf_device_map_auto else None,
        )
        try:
            model.to_bettertransformer()
        except Exception as e:
            logger.warning(f"Fail to call model.to_bettertransformer(), exception:\n{str(e)}")
        if not args.hf_device_map_auto:
            model.cuda()
        if model_name == "qwen":
            model.generation_config = GenerationConfig.from_pretrained(
                args.hf_model_dir, trust_remote_code=True
            )
        profiler.stop("load HF model")
        logger.info(f'Load HF model takes: {profiler.elapsed_time_in_sec("load HF model")} sec')

        datapoint = dataset[0:1]
        output, *_ = eval_hf(
            datapoint,
            eval_task=args.eval_task,
            eval_ppl=args.eval_ppl,
            add_special_tokens=args.add_special_tokens,
            min_input_length=args.min_input_length,
        )
        if runtime_rank == 0 and args.eval_task != "eval_context_ppl":
            logger.info("---------------------------------------------------------")
            logger.info("HF Generated : ")
            logger.info(f" Input : {datapoint[dataset_input_key]}")
            logger.info(f"\n Reference : {datapoint[dataset_output_key]}")
            logger.info(f"\n Output : {output}")
            logger.info("---------------------------------------------------------")

        ite_count = 0
        data_point_idx = 0
        total_output_token_count_hf = 0  # only valid for runtime_rank == 0
        while (data_point_idx < len(dataset)) and (ite_count < args.max_ite):
            if runtime_rank == 0:
                logger.debug(f"run data_point {data_point_idx} ~ {data_point_idx + max_batch_size}")
            datapoint = dataset[data_point_idx : (data_point_idx + max_batch_size)]

            profiler.start("hf")
            output_hf, token_list, curr_ppls_hf = eval_hf(
                datapoint,
                eval_task=args.eval_task,
                eval_ppl=args.eval_ppl,
                add_special_tokens=args.add_special_tokens,
                min_input_length=args.min_input_length,
            )
            profiler.stop("hf")
            if output_hf == []:
                data_point_idx += max_batch_size
                ite_count += 1
                continue
            if runtime_rank == 0:
                seq_lengths = [len(tokens) for tokens in token_list]
                total_output_token_count_hf += sum(seq_lengths)

                for beam_idx in range(num_beams):
                    for batch_idx in range(len(output_hf[beam_idx])):
                        metric_hf[beam_idx].add_batch(
                            predictions=[output_hf[beam_idx][batch_idx]],
                            references=[datapoint[dataset_output_key][batch_idx]],
                        )
                        if args.eval_ppl and args.batch_size == 1:
                            ppls_hf[beam_idx].append(curr_ppls_hf[batch_idx][beam_idx])
                if output_dir is not None:
                    for i in range(len(output_hf[0])):
                        for beam_idx in range(num_beams):
                            with (output_dir / "hf.out").open("a") as f:
                                f.write(
                                    f"[{data_point_idx + i}] [Beam {beam_idx}] {output_hf[beam_idx][i]}\n"
                                )

                logger.debug("-" * 100)
                logger.debug(f"Input : {datapoint[dataset_input_key]}")
                logger.debug(f"HF Output: {output_hf}")
                logger.debug(f"Reference : {datapoint[dataset_output_key]}")

            data_point_idx += max_batch_size
            ite_count += 1
        del model

    if runtime_rank == 0:
        if test_trt_llm:
            np.random.seed(0)  # rouge score use sampling to compute the score
            logger.info(
                f'TensorRT-LLM (total latency: {profiler.elapsed_time_in_sec("tensorrt_llm")} sec)'
            )
            logger.info(f"TensorRT-LLM (total output tokens: {total_output_token_count_trt_llm})")
            logger.info(
                "TensorRT-LLM (tokens per second:"
                f" {total_output_token_count_trt_llm / profiler.elapsed_time_in_sec('tensorrt_llm')})"
            )
            for beam_idx in range(num_beams):
                logger.info(f"TensorRT-LLM beam {beam_idx} result")
                computed_metrics_tensorrt_llm = metric_tensorrt_llm[beam_idx].compute()
                if args.eval_task != "eval_context_ppl":
                    for key in computed_metrics_tensorrt_llm.keys():
                        logger.info(f"  {key} : {computed_metrics_tensorrt_llm[key]*100}")

                if args.check_accuracy and beam_idx == 0 and args.eval_task != "eval_context_ppl":
                    assert (
                        computed_metrics_tensorrt_llm["rouge1"] * 100
                        > args.tensorrt_llm_rouge1_threshold
                    )
                if args.eval_ppl:
                    logger.info(f"  Per-token perplexity: {np.mean(ppls_trt_llm[beam_idx])}")
                    if args.check_accuracy and beam_idx == 0:
                        avg_ppl = np.mean(ppls_trt_llm[beam_idx])
                        assert avg_ppl < args.tensorrt_llm_ppl_threshold, (
                            f"[FAILED] average PPL ({avg_ppl}) is larger than "
                            f"threshold ({args.tensorrt_llm_ppl_threshold})"
                        )
        if test_hf:
            np.random.seed(0)  # rouge score use sampling to compute the score
            logger.info(f'Hugging Face (total latency: {profiler.elapsed_time_in_sec("hf")} sec)')
            logger.info(f"Hugging Face (total output tokens: {total_output_token_count_hf})")
            logger.info(
                f'Hugging Face (tokens per second: {total_output_token_count_hf / profiler.elapsed_time_in_sec("hf")})'
            )

            for beam_idx in range(num_beams):
                logger.info(f"HF beam {beam_idx} result")
                computed_metrics_hf = metric_hf[beam_idx].compute()
                if args.eval_task != "eval_context_ppl":
                    for key in computed_metrics_hf.keys():
                        logger.info(f"  {key} : {computed_metrics_hf[key]*100}")
                if args.eval_ppl and args.batch_size == 1:
                    logger.info(f"  Per-token perplexity: {np.mean(ppls_hf[beam_idx])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_hf", action="store_true")
    parser.add_argument("--test_trt_llm", action="store_true")
    parser.add_argument(
        "--eval_task",
        type=str,
        default="summarize",
        choices=["summarize", "summarize_long", "code_completion", "eval_context_ppl"],
    )
    parser.add_argument("--check_accuracy", action="store_true")
    parser.add_argument("--tensorrt_llm_rouge1_threshold", type=float, default=15.0)
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--tensorrt_llm_ppl_threshold", type=float, default=15.0)
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="The local directory of the dataset for evaluation; "
        "will download the dataset from huggingface hub if not specified.",
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=None,
        help="The local cache directory for dataset; "
        "will use `~/.cache/huggingface/datasets` if not specified.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_ite", type=int, default=20)
    parser.add_argument("--output_len", type=int, default=100)
    parser.add_argument("--max_input_length", type=int, default=923)
    parser.add_argument(
        "--min_input_length",
        type=int,
        default=0,
        help="skip the sentences which are shorter than min_input_length.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where to save output sentences. 'trtllm.out' for "
        "TensorRT-LLM outputs, and 'hf.out' for HF outputs.  If None, do not "
        "save outputs.",
    )
    parser.add_argument(
        "--rouge_dir",
        default=None,
        type=str,
        help="evaluate.load('rouge') will attempt to pull rouge package from HF. "
        "Use cached rouge can avoid network outage of host or HF.",
    )
    parser = add_common_args(parser)
    args = parser.parse_args()

    main(args)
