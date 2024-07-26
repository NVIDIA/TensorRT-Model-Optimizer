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

from typing import Any, Dict, Optional, Tuple, Union

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput


def cachecrossattnupblock2d_forward(
    self,
    hidden_states: torch.FloatTensor,
    res_hidden_states_0: torch.FloatTensor,
    res_hidden_states_1: torch.FloatTensor,
    res_hidden_states_2: torch.FloatTensor,
    temb: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    upsample_size: Optional[int] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
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
    temb: Optional[torch.FloatTensor] = None,
    upsample_size: Optional[int] = None,
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
    timestep: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor,
    class_labels: Optional[torch.Tensor] = None,
    timestep_cond: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    mid_block_additional_residual: Optional[torch.Tensor] = None,
    down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True,
) -> Union[UNet2DConditionOutput, Tuple]:

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
                if "res_samples_2" in down_results.keys():
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
        else:
            if hasattr(self, "use_trt_infer") and self.use_trt_infer:
                feed_dict = {"hidden_states": sample, "temb": emb}
                down_results = self.engines[f"down_blocks.{i}"](feed_dict, self.cuda_stream)
                sample = down_results["sample"]
                res_samples_0 = down_results["res_samples_0"]
                res_samples_1 = down_results["res_samples_1"]
                if "res_samples_2" in down_results.keys():
                    res_samples_2 = down_results["res_samples_2"]
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

        if hasattr(self, "use_trt_infer") and self.use_trt_infer:
            down_block_res_samples += (
                res_samples_0,
                res_samples_1,
            )
            if "res_samples_2" in down_results.keys():
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
        else:
            if hasattr(self, "use_trt_infer") and self.use_trt_infer:
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
