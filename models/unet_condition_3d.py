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
#
# ------------------------------------------------------------
# Modifications by Qi Pang (2026)
# Description: Extended to conditional 3D UNet architecture.
#   - Extended all 2D conv/attention blocks to 3D for seismic data (D, H, W)
# Original source: https://github.com/huggingface/diffusers
# ------------------------------------------------------------

import torch
from torch import nn

from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import apply_freeu

from diffusers.models.modeling_utils import ModelMixin

from diffusers.models.activations import get_activation
from diffusers.models.embeddings import SinusoidalPositionalEmbedding, Timesteps, TimestepEmbedding, LabelEmbedding, GaussianFourierProjection

from models.attention import AxialAttention_2
from models.blocks import ResnetBlock3D, Downsample3D, Upsample3D


@dataclass
class MyUnet3DModelOutput(BaseOutput):
    """
    The output of [`MyUNet3DCondition`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, depth, height, width)`)
    """

    sample: torch.Tensor = None


class MyAttnDownBlock3D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            attention_head_dim: int = 8,
            cross_attention_dim: int = 1280,
            output_scale_factor: float = 1.0,
            downsample_padding: int = 1,
            add_downsample: bool = True,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                AxialAttention_2(
                    out_channels,
                    attention_head_dim,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
            additional_residuals: Optional[torch.Tensor] = None,
    ):
        output_states = ()

        blocks = list(zip(self.resnets, self.attentions))

        for i, (resnet, attn) in enumerate(blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    added_cond_kwargs=added_cond_kwargs, )
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    added_cond_kwargs=added_cond_kwargs, )

            # apply additional residuals to the output of the last pair of resnet and attention blocks
            if i == len(blocks) - 1 and additional_residuals is not None:
                hidden_states = hidden_states + additional_residuals

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class DownBlock3D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            output_scale_factor: float = 1.0,
            add_downsample: bool = True,
            downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
            self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None):

        output_states = ()

        for resnet in self.resnets:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class MyAttnMidBlock3D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            temb_channels: int,
            out_channels: Optional[int] = None,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_groups_out: Optional[int] = None,
            resnet_pre_norm: bool = True,
            attention_head_dim: int = 8,
            cross_attention_dim: int = 1280,
            output_scale_factor: float = 1.0,
            downsample_padding: int = 1,
            add_downsample: bool = True,
            add_attention: bool = True,
    ):
        super().__init__()

        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_attention = add_attention

        self.has_cross_attention = True
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        resnet_groups_out = resnet_groups_out or resnet_groups

        # there is always at least one resnet
        resnets = [
            ResnetBlock3D(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                groups_out=resnet_groups_out,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        for i in range(num_layers):
            if self.add_attention:
                attentions.append(
                    AxialAttention_2(
                        out_channels,
                        attention_head_dim,
                    )
                )
            resnets.append(
                ResnetBlock3D(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups_out,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
            additional_residuals: Optional[torch.Tensor] = None,
    ):

        hidden_states = self.resnets[0](hidden_states, temb)
        if self.add_attention:
            for attn, resnet in zip(self.attentions, self.resnets[1:]):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = attn(
                        hidden_states,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        added_cond_kwargs=added_cond_kwargs, )
                    hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
                else:
                    hidden_states = attn(
                        hidden_states,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        added_cond_kwargs=added_cond_kwargs, )
                    hidden_states = resnet(hidden_states, temb)
        else:
            for resnet in self.resnets[1:]:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
                else:
                    hidden_states = resnet(hidden_states, temb)
                    
        return hidden_states


class MyAttnUpBlock3D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            prev_output_channel: int,
            temb_channels: int,
            resolution_idx: Optional[int] = None,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            attention_head_dim: int = 8,
            cross_attention_dim: int = 1280,
            output_scale_factor: float = 1.0,
            add_upsample: bool = True,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

            attentions.append(
                AxialAttention_2(
                    out_channels,
                    attention_head_dim,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample3D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
            self,
            hidden_states: torch.Tensor,
            res_hidden_states_tuple: Tuple[torch.Tensor, ...],
            temb: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            upsample_size: Optional[int] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        is_freeu_enabled = (
                getattr(self, "s1", None)
                and getattr(self, "s2", None)
                and getattr(self, "b1", None)
                and getattr(self, "b2", None)
        )

        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: Only operate on the first two stages
            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    added_cond_kwargs=added_cond_kwargs, )
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    added_cond_kwargs=added_cond_kwargs, )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class UpBlock3D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            prev_output_channel: int,
            out_channels: int,
            temb_channels: int,
            resolution_idx: Optional[int] = None,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            output_scale_factor: float = 1.0,
            add_upsample: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample3D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
            self,
            hidden_states: torch.Tensor,
            res_hidden_states_tuple: Tuple[torch.Tensor, ...],
            temb: Optional[torch.Tensor] = None,
            upsample_size: Optional[int] = None,
            *args,
            **kwargs,
    ) -> torch.Tensor:

        is_freeu_enabled = (
                getattr(self, "s1", None)
                and getattr(self, "s2", None)
                and getattr(self, "b1", None)
                and getattr(self, "b2", None)
        )

        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: Only operate on the first two stages
            if is_freeu_enabled:
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


def get_up_block(
        up_block_type: str,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        add_upsample: bool,
        resnet_eps: float,
        resnet_act_fn: str,
        resnet_groups: Optional[int] = None,
        attention_head_dim: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        resolution_idx: Optional[int] = None,
        resnet_time_scale_shift: str = "default",
        resnet_out_scale_factor: float = 1.0,
        dropout: float = 0.0,
):
    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock3D":
        return UpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "CrossAttnUpBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock2D")
        return MyAttnUpBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            num_layers=num_layers,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_head_dim=attention_head_dim,
            cross_attention_dim=cross_attention_dim,
        )
    raise ValueError(f"{up_block_type} does not exist.")


def get_down_block(
        down_block_type: str,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        add_downsample: bool,
        resnet_eps: float,
        resnet_act_fn: str,
        resnet_groups: Optional[int] = None,
        attention_head_dim: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        downsample_padding: Optional[int] = None,
        resnet_time_scale_shift: str = "default",
        dropout: float = 0.0,
):
    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownBlock3D":
        return DownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "CrossAttnDownBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim and num_frames must be specified for CrossAttnDownBlock2D")
        return MyAttnDownBlock3D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            num_layers=num_layers,
            resnet_eps=resnet_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            attention_head_dim=attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            downsample_padding=downsample_padding,
            add_downsample=add_downsample,
        )
    raise ValueError(f"{down_block_type} does not exist.")


class MyUNet3DCondition(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
            in_channels: int = 4,
            out_channels: int = 4,
            flip_sin_to_cos: bool = True,
            freq_shift: int = 0,
            down_block_types: Tuple[str] = (
                    "CrossAttnDownBlock3D",
                    "CrossAttnDownBlock3D",
                    "CrossAttnDownBlock3D",
                    "DownBlock3D",
            ),
            up_block_types: Tuple[str] = ("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
            add_mid_attn: bool = True, 
            block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
            layers_per_block: Union[int, Tuple[int]] = 2,
            downsample_padding: int = 1,
            mid_block_scale_factor: float = 1,
            dropout: float = 0.0,
            act_fn: str = "silu",
            norm_num_groups: Optional[int] = 32,
            norm_eps: float = 1e-5,

            attention_head_dim: Optional[int] = 8,
            # there cross_attention is never used just use axial attention
            cross_attention_dim: Union[int, Tuple[int]] = 4,

            encoder_hid_dim: Optional[int] = None,
            class_embed_type: Optional[str] = None,
            class_embeddings_concat: Optional[bool] = False,

            resnet_time_scale_shift: str = "default",
            time_embedding_dim=None,
            timestep_post_act: Optional[str] = None,
            time_cond_proj_dim: Optional[int] = None,
            conv_in_kernel: int = 3,
            conv_out_kernel: int = 3,
    ):
        super().__init__()

        # input 
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv3d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        # time 
        time_embed_dim = time_embedding_dim or block_out_channels[0] * 4
        # freq_i = torch.exp(torch.tensor(-math.log(10000.0) * i / (dim//2)))
        # time_embed[b][i] = torch.cos(timestep*freq_i)
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

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if encoder_hid_dim is not None:
            self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
        else:
            self.encoder_hid_proj = None

        if class_embeddings_concat:
            # The time embeddings are concatenated with the class embeddings. The dimension of the
            # time embeddings passed to the down, middle, and up blocks is twice the dimension of the
            # regular time embeddings
            blocks_time_embed_dim = time_embed_dim * 2
        else:
            blocks_time_embed_dim = time_embed_dim
            
        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)    

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            # do not add downsample in the final block
            is_final_block = i == len(block_out_channels) - 1
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,  # in final block is false
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

        # mid always attn
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

        # count how many layers upsample the images
        self.num_upsamplers = 0
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=reversed_attention_head_dim[i] if reversed_attention_head_dim[i] is not None else output_channel,
                cross_attention_dim=reversed_cross_attention_dim[i],
                resolution_idx=i,
                resnet_time_scale_shift=resnet_time_scale_shift,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)

            # out
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
            )

            self.conv_act = get_activation(act_fn)

        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv3d(
            block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding
        )

    def get_time_embed(
            self, sample: torch.Tensor, timestep: Optional[torch.LongTensor],
            timestep_cond: Optional[torch.LongTensor]):
        timesteps = timestep
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        return emb

    def get_class_embed(
            self, sample: torch.Tensor, class_labels: Optional[torch.Tensor]):
        class_emb = None
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)
        return class_emb

    def forward(self,
                sample: torch.Tensor,
                timestep: Optional[torch.LongTensor] = None,
                encoder_hidden_states: torch.Tensor = None,
                class_labels: Optional[torch.Tensor] = None,
                timestep_cond: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
                down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
                mid_block_additional_residual: Optional[torch.Tensor] = None,
                ):

        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.        
        default_overall_up_factor = 2 ** self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        # prepare for 2d
        # b, t, c, h, w = sample.shape
        # sample = sample.view(-1, c, h, w)

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break
            
        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        
        # 1. time
        emb = self.get_time_embed(sample=sample,
                                  timestep=timestep, timestep_cond=timestep_cond)

        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)

        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

                # get_aug_embed will be considered later
        if self.encoder_hid_proj:
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

        # 2. pre-process
        sample = self.conv_in(sample)

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
        
        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples        
        
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
                
        if is_controlnet:
            sample = sample + mid_block_additional_residual
            
        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    added_cond_kwargs=added_cond_kwargs,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        # sample = sample.view(b, t, c, h, w)
        return MyUnet3DModelOutput(sample=sample)


if __name__ == "__main__":
    block_out_channels = (16, 32, 64)
    down_block_types = ("DownBlock3D",
                        "DownBlock3D",
                        "CrossAttnDownBlock3D"
                        )
    up_block_types = ("CrossAttnUpBlock3D",
                      "UpBlock3D",
                      "UpBlock3D")

    b, c, d, h, w = 1, 4, 32, 32, 32

    sample = torch.randn(b, c, d, h, w)
    timestep = torch.tensor([7])
    class_labels = torch.randn(b)
    # sample = sample.view(b*t, c, h, w)
    # encoder_hidden_states = torch.randn(b, t, c, h, w)
    # cond_positions = torch.randn(b, t)     
    # added_cond_kwargs = {
    # "cond_positions": cond_positions,
    # }  
    # added_cond_kwargs=None

    D = MyUNet3DCondition(in_channels=c,
                          out_channels=c,
                          block_out_channels=block_out_channels,
                          down_block_types=down_block_types,
                          up_block_types=up_block_types,
                          layers_per_block=2,
                          attention_head_dim=8,
                          norm_num_groups=8,
                          # cross_attention_dim=t,
                          # add_mid_attn=False
                          )

    hidden = D(sample, timestep,
               # class_labels=class_labels,
               # encoder_hidden_states=encoder_hidden_states,
               # added_cond_kwargs=added_cond_kwargs
               )

    out = hidden.sample
