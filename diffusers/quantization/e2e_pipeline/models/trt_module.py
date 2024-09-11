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

from typing import Any

KWDS_MAPPING = {
    "sdxl-1.0": {
        "sample": "latent_model_input",
    },
}


class TrTBackBone:
    def __init__(self, config, engine, stream, use_cuda_graph=False, pipe_name: str = None) -> None:
        self.config = config
        self.engine = engine
        self.stream = stream
        self.pipe_name = pipe_name
        self.use_cuda_graph = use_cuda_graph

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.pipe_name == "sdxl-1.0":
            feed_dict = {
                "sample": args[0],
                "timestep": args[1],
                "encoder_hidden_states": kwds["encoder_hidden_states"],
                "text_embeds": kwds["added_cond_kwargs"]["text_embeds"],
                "time_ids": kwds["added_cond_kwargs"]["time_ids"],
            }
        elif self.pipe_name == "sd3-medium":
            feed_dict = {
                "hidden_states": kwds["hidden_states"],
                "timestep": kwds["timestep"],
                "encoder_hidden_states": kwds["encoder_hidden_states"],
                "pooled_projections": kwds["pooled_projections"],
            }
        elif self.pipe_name == "flux-dev":
            feed_dict = {
                "hidden_states": kwds["hidden_states"],
                "img_ids": kwds["img_ids"],
                "encoder_hidden_states": kwds["encoder_hidden_states"],
                "txt_ids": kwds["txt_ids"],
                "timestep": kwds["timestep"],
                "pooled_projections": kwds["pooled_projections"],
                "guidance": kwds["guidance"],
            }
        else:
            NotImplementedError
        inf_results = self.engine(
            feed_dict=feed_dict, stream=self.stream, use_cuda_graph=self.use_cuda_graph
        )
        if self.pipe_name == "sdxl-1.0":
            final_results = (inf_results["latent"],)
        elif self.pipe_name == "flux-dev":
            final_results = (inf_results["output"],)
        else:
            final_results = (inf_results["sample"],)
        return final_results
