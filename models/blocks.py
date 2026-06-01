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
# Description: Extended Diffusers' ResnetBlock2D, Downsample2D, and
#   Upsample2D to their 3D counterparts (ResnetBlock3D, Downsample3D,
#   Upsample3D) for volumetric seismic data processing.
# Original source: https://github.com/huggingface/diffusers
# ------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from diffusers.models.activations import get_activation
from diffusers.utils.import_utils import is_torch_version


class ResnetBlock3D(nn.Module):
    def __init__(
            self,
            *,
            in_channels: int,
            out_channels: Optional[int] = None,
            conv_shortcut: bool = False,
            dropout: float = 0.0,
            temb_channels: int = 512,
            groups: int = 32,
            groups_out: Optional[int] = None,
            pre_norm: bool = True,
            eps: float = 1e-6,
            non_linearity: str = "swish",
            skip_time_act: bool = False,
            time_embedding_norm: str = "default",  # default, scale_shift,
            kernel: Optional[torch.Tensor] = None,
            output_scale_factor: float = 1.0,
            use_in_shortcut: Optional[bool] = None,
            conv_shortcut_bias: bool = True,
            conv_3d_out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = nn.Linear(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = nn.Linear(temb_channels, 2 * out_channels)
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        conv_3d_out_channels = conv_3d_out_channels or out_channels
        self.conv2 = nn.Conv3d(out_channels, conv_3d_out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity = get_activation(non_linearity)

        self.use_in_shortcut = self.in_channels != conv_3d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv3d(
                in_channels,
                conv_3d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)
        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None, None]  
        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError(
                    f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                )
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor.contiguous())

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class Downsample3D(nn.Module):
    def __init__(
            self,
            channels: int,
            use_conv: bool = False,
            out_channels: Optional[int] = None,
            groups: int = 32,
            padding: int = 1,
            name: str = "conv",
            kernel_size=3,
            norm_type=None,
            eps: float = 1e-6,
            bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if norm_type == "gn_norm":
            self.norm = torch.nn.GroupNorm(num_groups=groups, num_channels=channels, eps=eps, affine=True)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        if use_conv:
            conv = nn.Conv3d(
                self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool3d(kernel_size=stride, stride=stride)

        self.conv = conv

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states)

        return hidden_states


class Upsample3D(nn.Module):
    def __init__(
            self,
            channels: int,
            use_conv: bool = False,
            use_conv_transpose: bool = False,
            out_channels: Optional[int] = None,
            groups: int = 32,
            name: str = "conv",
            kernel_size: Optional[int] = None,
            padding=1,
            norm_type=None,
            eps: float = 1e-6,
            bias=True,
            interpolate=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate

        if norm_type == "gn_norm":
            self.norm = torch.nn.GroupNorm(num_groups=groups, num_channels=channels, eps=eps, affine=True)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            conv = nn.ConvTranspose3d(
                channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = nn.Conv3d(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        self.conv = conv

    def forward(self, hidden_states: torch.Tensor, output_size: Optional[int] = None, *args, **kwargs) -> torch.Tensor:

        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16 until PyTorch 2.1
        # https://github.com/pytorch/pytorch/issues/86679#issuecomment-1783978767
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16 and is_torch_version("<", "2.1"):
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if self.interpolate:
            # upsample_nearest_nhwc also fails when the number of output elements is large
            # https://github.com/pytorch/pytorch/issues/141831
            scale_factor = (
                2 if output_size is None else max([f / s for f, s in zip(output_size, hidden_states.shape[-2:])])
            )
            if hidden_states.numel() * scale_factor > pow(2, 31):
                hidden_states = hidden_states.contiguous()

            if output_size is None:
                hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            else:
                hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # Cast back to original dtype
        if dtype == torch.bfloat16 and is_torch_version("<", "2.1"):
            hidden_states = hidden_states.to(dtype)

        if self.use_conv:
            hidden_states = self.conv(hidden_states)

        return hidden_states


if __name__ == "__main__":
    b, c, d, h, w = 1, 32, 32, 32, 32
    sample = torch.randn(b, c, d, h, w)
    # net = ResnetBlock3D(in_channels=32, temb_channels=None)
    # net = Downsample3D(channels=32, use_conv=True, )  
    net = Upsample3D(channels=32, use_conv=True, )
    out = net(sample)
