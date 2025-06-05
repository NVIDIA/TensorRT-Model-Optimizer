# Adapted from
# https://github.com/huggingface/diffusers/blob/73acebb8cfbd1d2954cabe1af4185f9994e61917/src/diffusers/models/unets/unet_2d_condition.py#L1039-L1312
# https://github.com/huggingface/diffusers/blob/73acebb8cfbd1d2954cabe1af4185f9994e61917/src/diffusers/models/unets/unet_2d_blocks.py#L2482-L2564
# https://github.com/huggingface/diffusers/blob/73acebb8cfbd1d2954cabe1af4185f9994e61917/src/diffusers/models/unets/unet_2d_blocks.py#L2617-L2679

# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

from typing import Any

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput


def cachecrossattnupblock2d_forward(
    self,
    hidden_states: torch.FloatTensor,
    res_hidden_states_0: torch.FloatTensor,
    res_hidden_states_1: torch.FloatTensor,
    res_hidden_states_2: torch.FloatTensor,
    temb: torch.FloatTensor | None = None,
    encoder_hidden_states: torch.FloatTensor | None = None,
    cross_attention_kwargs: dict[str, Any] | None = None,
    upsample_size: int | None = None,
    attention_mask: torch.FloatTensor | None = None,
    encoder_attention_mask: torch.FloatTensor | None = None,
) -> torch.FloatTensor:
    res_hidden_states_tuple = (res_hidden_states_0, res_hidden_states_1, res_hidden_states_2)
    for resnet, attn in zip(self.resnets, self.attentions):
        # pop res hidden states
        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]

        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        hidden_states = resnet(hidden_states, temb)
        hidden_states = attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )[0]

    if self.upsamplers is not None:
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states, upsample_size)

    return hidden_states


def cacheupblock2d_forward(
    self,
    hidden_states: torch.FloatTensor,
    res_hidden_states_0: torch.FloatTensor,
    res_hidden_states_1: torch.FloatTensor,
    res_hidden_states_2: torch.FloatTensor,
    temb: torch.FloatTensor | None = None,
    upsample_size: int | None = None,
) -> torch.FloatTensor:
    res_hidden_states_tuple = (res_hidden_states_0, res_hidden_states_1, res_hidden_states_2)
    for resnet in self.resnets:
        # pop res hidden states
        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]

        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

        hidden_states = resnet(hidden_states, temb)

    if self.upsamplers is not None:
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states, upsample_size)

    return hidden_states


def cacheunet_forward(
    self,
    sample: torch.FloatTensor,
    timestep: torch.Tensor | float | int,
    encoder_hidden_states: torch.Tensor,
    class_labels: torch.Tensor | None = None,
    timestep_cond: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    cross_attention_kwargs: dict[str, Any] | None = None,
    added_cond_kwargs: dict[str, torch.Tensor] | None = None,
    down_block_additional_residuals: tuple[torch.Tensor] | None = None,
    mid_block_additional_residual: torch.Tensor | None = None,
    down_intrablock_additional_residuals: tuple[torch.Tensor] | None = None,
    encoder_attention_mask: torch.Tensor | None = None,
    return_dict: bool = True,
) -> UNet2DConditionOutput | tuple:
    # 1. time
    t_emb = self.get_time_embed(sample=sample, timestep=timestep)
    emb = self.time_embedding(t_emb, timestep_cond)
    aug_emb = None

    aug_emb = self.get_aug_embed(
        emb=emb,
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs=added_cond_kwargs,
    )

    emb = emb + aug_emb if aug_emb is not None else emb

    encoder_hidden_states = self.process_encoder_hidden_states(
        encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )

    # 2. pre-process
    sample = self.conv_in(sample)

    if hasattr(self, "_export_precess_onnx") and self._export_precess_onnx:
        return (
            sample,
            encoder_hidden_states,
            emb,
        )

    down_block_res_samples = (sample,)
    for i, downsample_block in enumerate(self.down_blocks):
        if (
            hasattr(downsample_block, "has_cross_attention")
            and downsample_block.has_cross_attention
        ):
            if hasattr(self, "use_trt_infer") and self.use_trt_infer:
                feed_dict = {
                    "hidden_states": sample,
                    "temb": emb,
                    "encoder_hidden_states": encoder_hidden_states,
                }
                down_results = self.engines[f"down_blocks.{i}"](feed_dict, self.cuda_stream)
                sample = down_results["sample"]
                res_samples_0 = down_results["res_samples_0"]
                res_samples_1 = down_results["res_samples_1"]
                if "res_samples_2" in down_results:
                    res_samples_2 = down_results["res_samples_2"]
            else:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
        elif hasattr(self, "use_trt_infer") and self.use_trt_infer:
            feed_dict = {"hidden_states": sample, "temb": emb}
            down_results = self.engines[f"down_blocks.{i}"](feed_dict, self.cuda_stream)
            sample = down_results["sample"]
            res_samples_0 = down_results["res_samples_0"]
            res_samples_1 = down_results["res_samples_1"]
            if "res_samples_2" in down_results:
                res_samples_2 = down_results["res_samples_2"]
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

        if hasattr(self, "use_trt_infer") and self.use_trt_infer:
            down_block_res_samples += (
                res_samples_0,
                res_samples_1,
            )
            if "res_samples_2" in down_results:
                down_block_res_samples += (res_samples_2,)
        else:
            down_block_res_samples += res_samples

    if hasattr(self, "use_trt_infer") and self.use_trt_infer:
        feed_dict = {
            "hidden_states": sample,
            "temb": emb,
            "encoder_hidden_states": encoder_hidden_states,
        }
        mid_results = self.engines["mid_block"](feed_dict, self.cuda_stream)
        sample = mid_results["sample"]
    else:
        sample = self.mid_block(
            sample,
            emb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            encoder_attention_mask=encoder_attention_mask,
        )

    # 5. up
    for i, upsample_block in enumerate(self.up_blocks):
        res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

        if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
            if hasattr(self, "use_trt_infer") and self.use_trt_infer:
                feed_dict = {
                    "hidden_states": sample,
                    "res_hidden_states_0": res_samples[0],
                    "res_hidden_states_1": res_samples[1],
                    "res_hidden_states_2": res_samples[2],
                    "temb": emb,
                    "encoder_hidden_states": encoder_hidden_states,
                }
                up_results = self.engines[f"up_blocks.{i}"](feed_dict, self.cuda_stream)
                sample = up_results["sample"]
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_0=res_samples[0],
                    res_hidden_states_1=res_samples[1],
                    res_hidden_states_2=res_samples[2],
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
        elif hasattr(self, "use_trt_infer") and self.use_trt_infer:
            feed_dict = {
                "hidden_states": sample,
                "res_hidden_states_0": res_samples[0],
                "res_hidden_states_1": res_samples[1],
                "res_hidden_states_2": res_samples[2],
                "temb": emb,
            }
            up_results = self.engines[f"up_blocks.{i}"](feed_dict, self.cuda_stream)
            sample = up_results["sample"]
        else:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_0=res_samples[0],
                res_hidden_states_1=res_samples[1],
                res_hidden_states_2=res_samples[2],
            )

    # 6. post-process
    if self.conv_norm_out:
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
    sample = self.conv_out(sample)

    if not return_dict:
        return (sample,)

    return UNet2DConditionOutput(sample=sample)
