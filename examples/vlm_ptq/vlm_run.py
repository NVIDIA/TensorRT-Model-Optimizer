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

import os

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm import logger
from tensorrt_llm.runtime import MultimodalModelRunner
from utils import parse_arguments


def print_result(model, input_text, output_text, args):
    logger.info("---------------------------------------------------------")
    if model.model_type != "nougat":
        logger.info(f"\n[Q] {input_text}")
    for i in range(len(output_text)):
        logger.info(f"\n[A]: {output_text[i]}")

    if args.num_beams == 1:
        output_ids = model.tokenizer(output_text[0][0], add_special_tokens=False)["input_ids"]
        logger.info(f"Generated {len(output_ids)} tokens")

    if args.check_accuracy:
        if model.model_type != "nougat":
            if model.model_type == "vila":
                if len(args.image_path.split(args.path_sep)) == 1:
                    assert (
                        output_text[0][0].lower()
                        == "the image captures a bustling city intersection teeming with life. "
                        "from the perspective of a car's dashboard camera, we see"
                    )
            elif model.model_type == "fuyu":
                assert output_text[0][0].lower() == "4"
            elif model.model_type == "pix2struct":
                assert (
                    "characteristic | cat food, day | cat food, wet | cat treats"
                    in output_text[0][0].lower()
                )
            elif model.model_type in ["blip2", "neva", "phi-3-vision", "llava_next"]:
                assert "singapore" in output_text[0][0].lower()
            elif model.model_type == "video-neva":
                assert "robot" in output_text[0][0].lower()
            elif model.model_type == "kosmos-2":
                assert "snowman" in output_text[0][0].lower()
            else:
                assert output_text[0][0].lower() == "singapore"

    if args.run_profiling:

        def msec_per_batch(name):
            return 1000 * profiler.elapsed_time_in_sec(name) / args.profiling_iterations

        logger.info("Latencies per batch (msec)")
        logger.info("TRT vision encoder: %.1f" % (msec_per_batch("Vision")))
        logger.info("TRTLLM LLM generate: %.1f" % (msec_per_batch("LLM")))
        logger.info("Multimodal generate: %.1f" % (msec_per_batch("Generate")))

    logger.info("---------------------------------------------------------")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_arguments("run")
    logger.set_level(args.log_level)

    model = MultimodalModelRunner(args)
    raw_image = model.load_test_data()
    if args.input_text:
        input_text = args.input_text.split(args.prompt_sep)
        if len(input_text) == 1:
            input_text = input_text[0]
    else:
        input_text = args.input_text

    num_iters = args.profiling_iterations if args.run_profiling else 1
    for _ in range(num_iters):
        input_text, output_text = model.run(input_text, raw_image, args.max_new_tokens)
    runtime_rank = tensorrt_llm.mpi_rank()
    if runtime_rank == 0:
        print_result(model, input_text, output_text, args)
