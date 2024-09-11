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
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

sd3_common_transformer_block_config = {
    "dummy_input": {
        "hidden_states": (2, 4096, 1536),
        "encoder_hidden_states": (2, 333, 1536),
        "temb": (2, 1536),
    },
    "output_names": ["encoder_hidden_states_out", "hidden_states_out"],
    "dynamic_axes": {
        "hidden_states": {0: "batch_size"},
        "encoder_hidden_states": {0: "batch_size"},
        "temb": {0: "steps"},
    },
}

ONNX_CONFIG = {
    UNet2DConditionModel: {
        "down_blocks.0": {
            "dummy_input": {
                "hidden_states": (2, 320, 128, 128),
                "temb": (2, 1280),
            },
            "output_names": ["sample", "res_samples_0", "res_samples_1", "res_samples_2"],
            "dynamic_axes": {
                "hidden_states": {0: "batch_size"},
                "temb": {0: "steps"},
            },
        },
        "down_blocks.1": {
            "dummy_input": {
                "hidden_states": (2, 320, 64, 64),
                "temb": (2, 1280),
                "encoder_hidden_states": (2, 77, 2048),
            },
            "output_names": ["sample", "res_samples_0", "res_samples_1", "res_samples_2"],
            "dynamic_axes": {
                "hidden_states": {0: "batch_size"},
                "temb": {0: "steps"},
                "encoder_hidden_states": {0: "batch_size"},
            },
        },
        "down_blocks.2": {
            "dummy_input": {
                "hidden_states": (2, 640, 32, 32),
                "temb": (2, 1280),
                "encoder_hidden_states": (2, 77, 2048),
            },
            "output_names": ["sample", "res_samples_0", "res_samples_1"],
            "dynamic_axes": {
                "hidden_states": {0: "batch_size"},
                "temb": {0: "steps"},
                "encoder_hidden_states": {0: "batch_size"},
            },
        },
        "mid_block": {
            "dummy_input": {
                "hidden_states": (2, 1280, 32, 32),
                "temb": (2, 1280),
                "encoder_hidden_states": (2, 77, 2048),
            },
            "output_names": ["sample"],
            "dynamic_axes": {
                "hidden_states": {0: "batch_size"},
                "temb": {0: "steps"},
                "encoder_hidden_states": {0: "batch_size"},
            },
        },
        "up_blocks.0": {
            "dummy_input": {
                "hidden_states": (2, 1280, 32, 32),
                "res_hidden_states_0": (2, 640, 32, 32),
                "res_hidden_states_1": (2, 1280, 32, 32),
                "res_hidden_states_2": (2, 1280, 32, 32),
                "temb": (2, 1280),
                "encoder_hidden_states": (2, 77, 2048),
            },
            "output_names": ["sample"],
            "dynamic_axes": {
                "hidden_states": {0: "batch_size"},
                "temb": {0: "steps"},
                "encoder_hidden_states": {0: "batch_size"},
                "res_hidden_states_0": {0: "batch_size"},
                "res_hidden_states_1": {0: "batch_size"},
                "res_hidden_states_2": {0: "batch_size"},
            },
        },
        "up_blocks.1": {
            "dummy_input": {
                "hidden_states": (2, 1280, 64, 64),
                "res_hidden_states_0": (2, 320, 64, 64),
                "res_hidden_states_1": (2, 640, 64, 64),
                "res_hidden_states_2": (2, 640, 64, 64),
                "temb": (2, 1280),
                "encoder_hidden_states": (2, 77, 2048),
            },
            "output_names": ["sample"],
            "dynamic_axes": {
                "hidden_states": {0: "batch_size"},
                "temb": {0: "steps"},
                "encoder_hidden_states": {0: "batch_size"},
                "res_hidden_states_0": {0: "batch_size"},
                "res_hidden_states_1": {0: "batch_size"},
                "res_hidden_states_2": {0: "batch_size"},
            },
        },
        "up_blocks.2": {
            "dummy_input": {
                "hidden_states": (2, 640, 128, 128),
                "res_hidden_states_0": (2, 320, 128, 128),
                "res_hidden_states_1": (2, 320, 128, 128),
                "res_hidden_states_2": (2, 320, 128, 128),
                "temb": (2, 1280),
            },
            "output_names": ["sample"],
            "dynamic_axes": {
                "hidden_states": {0: "batch_size"},
                "temb": {0: "steps"},
                "res_hidden_states_0": {0: "batch_size"},
                "res_hidden_states_1": {0: "batch_size"},
                "res_hidden_states_2": {0: "batch_size"},
            },
        },
    },
    SD3Transformer2DModel: {
        **{f"transformer_blocks.{i}": sd3_common_transformer_block_config for i in range(23)},
        "transformer_blocks.23": {
            "dummy_input": {
                "hidden_states": (2, 4096, 1536),
                "encoder_hidden_states": (2, 333, 1536),
                "temb": (2, 1536),
            },
            "output_names": ["hidden_states_out"],
            "dynamic_axes": {
                "hidden_states": {0: "batch_size"},
                "encoder_hidden_states": {0: "batch_size"},
                "temb": {0: "steps"},
            },
        },
    },
}
