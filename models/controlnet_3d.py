# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# ------------------------------------------------------------
# Modifications by Qi Pang (2026)
# Description: Extended ControlNet to 3D architecture for
#   volumetric seismic data processing.
#   Added ControlNetConditioningEmbedding3D: 3D conv embedding
#   for binary fault mask conditioning input.
# Original source: https://github.com/huggingface/diffusers
# ------------------------------------------------------------

import torch
from torch import nn
from torch.nn import functional as F

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from diffusers.utils import BaseOutput
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import Timesteps, TimestepEmbedding

from models.unet_condition_3d import get_down_block, MyAttnMidBlock3D


@dataclass
class ControlNetOutput(BaseOutput):
    """
    The output of [`ControlNetModel`].
    
    Args:
        down_block_res_samples:
            Downsample activations at each resolution, each of shape
            (B, C, D//r, H//r, W//r) where r is the resolution factor.
        mid_block_res_sample:
            Middle block activation of shape (B, C, D//r_max, H//r_max, W//r_max).
    """

    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ControlNetConditioningEmbedding(nn.Module):
    def __init__(
            self,
            conditioning_embedding_channels: int,
            conditioning_channels: int = 3,
            block_out_channels: Tuple[int, ...] = (16, 32, 96),
    ):
        super().__init__()

        self.conv_in = nn.Conv3d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv3d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv3d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv3d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


class MyControlNet3D(ModelMixin, ConfigMixin):
    """
    3D ControlNet for conditioning MyUNet3DCondition
    """

    @register_to_config
    def __init__(
            self,
            in_channels: int = 4,
            conditioning_channels: int = 3,
            flip_sin_to_cos: bool = True,
            freq_shift: int = 0,
            down_block_types: Tuple[str] = (
                    "CrossAttnDownBlock3D",
                    "CrossAttnDownBlock3D",
                    "CrossAttnDownBlock3D",
                    "DownBlock3D",
            ),
            add_mid_attn: bool = True,
            block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
            layers_per_block: int = 2,
            downsample_padding: int = 1,
            mid_block_scale_factor: float = 1,
            dropout: float = 0.0,
            act_fn: str = "silu",
            norm_num_groups: Optional[int] = 32,
            norm_eps: float = 1e-5,
            attention_head_dim: int = 8,
            cross_attention_dim: int = 4,
            encoder_hid_dim: Optional[int] = None,
            class_embed_type: Optional[str] = None,
            resnet_time_scale_shift: str = "default",
            time_embedding_dim: Optional[int] = None,
            timestep_post_act: Optional[str] = None,
            time_cond_proj_dim: Optional[int] = None,
            conv_in_kernel: int = 3,
            conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96),
            global_pool_conditions: bool = False,
    ):
        super().__init__()
        self.global_pool_conditions = global_pool_conditions

        # input 
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv3d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        # time 
        time_embed_dim = time_embedding_dim or block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )

        if class_embed_type is not None:
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, act_fn=act_fn)
        else:
            self.class_embedding = None

        if encoder_hid_dim is not None:
            self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
        else:
            self.encoder_hid_proj = None

            # controlnet
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
        )

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)
        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)
        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        blocks_time_embed_dim = time_embed_dim

        # down
        self.down_blocks = nn.ModuleList([])
        self.controlnet_down_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]
        controlnet_block = nn.Conv3d(output_channel, output_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                cross_attention_dim=cross_attention_dim[i],
                downsample_padding=downsample_padding,
                resnet_time_scale_shift=resnet_time_scale_shift,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

            for _ in range(layers_per_block[i]):
                controlnet_block = nn.Conv3d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

            if not is_final_block:
                controlnet_block = nn.Conv3d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

                # mid
        controlnet_block = nn.Conv3d(block_out_channels[-1], block_out_channels[-1], kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_mid_block = controlnet_block

        self.mid_block = MyAttnMidBlock3D(
            temb_channels=blocks_time_embed_dim,
            in_channels=block_out_channels[-1],
            dropout=dropout,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            output_scale_factor=mid_block_scale_factor,
            attention_head_dim=attention_head_dim[-1],
            cross_attention_dim=cross_attention_dim[-1],
            add_attention=add_mid_attn
        )

    @classmethod
    def from_unet(
            cls,
            unet,
            conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96),
            load_weights_from_unet: bool = True,
            conditioning_channels: int = 3,
    ):
        config = unet.config

        controlnet = cls(
            in_channels=config.in_channels,
            conditioning_channels=conditioning_channels,
            flip_sin_to_cos=config.flip_sin_to_cos,
            freq_shift=config.freq_shift,
            down_block_types=config.down_block_types,
            add_mid_attn=config.add_mid_attn,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
            downsample_padding=config.downsample_padding,
            mid_block_scale_factor=config.mid_block_scale_factor,
            dropout=config.dropout,
            act_fn=config.act_fn,
            norm_num_groups=config.norm_num_groups,
            norm_eps=config.norm_eps,
            attention_head_dim=config.attention_head_dim,
            cross_attention_dim=config.cross_attention_dim,
            encoder_hid_dim=config.encoder_hid_dim,
            class_embed_type=config.class_embed_type,
            resnet_time_scale_shift=config.resnet_time_scale_shift,
            time_embedding_dim=config.time_embedding_dim,
            timestep_post_act=config.timestep_post_act,
            time_cond_proj_dim=config.time_cond_proj_dim,
            conv_in_kernel=config.conv_in_kernel,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
        )

        if load_weights_from_unet:

            controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
            controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
            controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())

            if controlnet.class_embedding is not None:
                controlnet.class_embedding.load_state_dict(unet.class_embedding.state_dict())

            if controlnet.encoder_hid_proj is not None:
                controlnet.encoder_hid_proj.load_state_dict(unet.encoder_hid_proj.state_dict())

            for controlnet_block, unet_block in zip(controlnet.down_blocks, unet.down_blocks):
                controlnet_block.load_state_dict(unet_block.state_dict())

            controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())

        return controlnet

    def forward(
            self,
            sample: torch.Tensor,
            timestep: torch.LongTensor,
            encoder_hidden_states: torch.Tensor,
            controlnet_cond: torch.Tensor,
            conditioning_scale: float = 1.0,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ):

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided")
            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)
            emb = emb + class_emb

            # 2. pre-process
        sample = self.conv_in(sample)

        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        sample = sample + controlnet_cond

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:

            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    added_cond_kwargs=added_cond_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    added_cond_kwargs=added_cond_kwargs,
                )
            else:
                sample = self.mid_block(sample, emb)

        # 5. control net blocks
        controlnet_down_block_res_samples = ()

        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples
        mid_block_res_sample = self.controlnet_mid_block(sample)

        # 6. scaling
        down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
        mid_block_res_sample = mid_block_res_sample * conditioning_scale

        if self.config.global_pool_conditions:
            down_block_res_samples = [
                torch.mean(sample, dim=(2, 3, 4), keepdim=True) for sample in down_block_res_samples
            ]
            mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3, 4), keepdim=True)

        return ControlNetOutput(
            down_block_res_samples=down_block_res_samples, mid_block_res_sample=mid_block_res_sample
        )


if __name__ == "__main__":
    block_out_channels = (16, 32, 64)
    down_block_types = ("DownBlock3D",
                        "DownBlock3D",
                        "CrossAttnDownBlock3D"
                        )
    up_block_types = ("CrossAttnUpBlock3D",
                      "UpBlock3D",
                      "UpBlock3D")

    b, c, d, h, w = 1, 1, 128, 128, 128

    sample = torch.randn(b, c, d, h, w)

    net = ControlNetConditioningEmbedding(32, 1)

    out = net(sample)
