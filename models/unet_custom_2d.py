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
#   - Removed timestep embedding (deterministic regressor, not diffusion model)
#   - Adapted for seismic-to-impedance inversion task
# Original source: https://github.com/huggingface/diffusers
# ------------------------------------------------------------
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_2d_blocks import (
    UNetMidBlock2D,
    get_down_block,
    get_up_block,
)


@dataclass
class CustomUNet2DOutput:
    sample: torch.FloatTensor


class CustomUNet2D(ModelMixin, ConfigMixin):
    """Diffusers-style UNet without timestep embedding.

    Input  : (B, in_channels, H, W) — typically normalized seismic
    Output : (B, out_channels, H, W) — typically normalized impedance
    """
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        down_block_types: Tuple[str, ...] = (
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types: Tuple[str, ...] = (
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        block_out_channels: Tuple[int, ...] = (64, 128, 256, 512),
        layers_per_block: Union[int, Tuple[int, ...]] = 2,
        mid_block_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        attention_head_dim: Union[int, Tuple[int, ...]] = 8,
        use_out_tanh: bool = False,
    ):
        super().__init__()

        # ---------- checks ----------
        if len(down_block_types) != len(up_block_types):
            raise ValueError("down_block_types and up_block_types length must match.")
        if len(block_out_channels) != len(down_block_types):
            raise ValueError("block_out_channels length must match block types length.")

        n_blocks = len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = tuple([layers_per_block] * n_blocks)
        elif len(layers_per_block) != n_blocks:
            raise ValueError("layers_per_block tuple length must match block count.")

        if isinstance(attention_head_dim, int):
            attention_head_dim = tuple([attention_head_dim] * n_blocks)
        elif len(attention_head_dim) != n_blocks:
            raise ValueError("attention_head_dim tuple length must match block count.")

        # ---------- input ----------
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        # ---------- down ----------
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == n_blocks - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=None,  # no time embedding
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim[i],
                downsample_padding=downsample_padding,
            )
            self.down_blocks.append(down_block)

        # ---------- mid ----------
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=None,  # no time embedding
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            attention_head_dim=attention_head_dim[-1],
            resnet_groups=norm_num_groups,
        )

        # ---------- up ----------
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_attention_head_dim = list(reversed(attention_head_dim))

        prev_output_channel = reversed_block_out_channels[0]

        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == n_blocks - 1

            out_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, n_blocks - 1)]

            num_layers = reversed_layers_per_block[i] + 1

            up_block = get_up_block(
                up_block_type,
                num_layers=num_layers,
                in_channels=input_channel,
                out_channels=out_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=None,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=reversed_attention_head_dim[i],
            )
            self.up_blocks.append(up_block)
            prev_output_channel = out_channel

        # ---------- output ----------
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0],
            num_groups=norm_num_groups,
            eps=norm_eps,
        )
        self.conv_act = nn.SiLU() if act_fn == "silu" else nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)
        self.out_tanh = nn.Tanh() if use_out_tanh else nn.Identity()

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value
        self.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ):
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # in
        sample = self.conv_in(sample)

        # down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if self.training and self.gradient_checkpointing:
                sample, res_samples = torch.utils.checkpoint.checkpoint(
                    lambda s: downsample_block(hidden_states=s, temb=None),
                    sample,
                    use_reentrant=False,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=None)

            down_block_res_samples += res_samples

        # mid
        if self.training and self.gradient_checkpointing:
            sample = torch.utils.checkpoint.checkpoint(
                lambda s: self.mid_block(s, temb=None),
                sample,
                use_reentrant=False,
            )
        else:
            sample = self.mid_block(sample, temb=None)

        # up
        for upsample_block in self.up_blocks:
            n_resnets = len(upsample_block.resnets)
            res_samples = down_block_res_samples[-n_resnets:]
            down_block_res_samples = down_block_res_samples[:-n_resnets]

            if self.training and self.gradient_checkpointing:
                sample = torch.utils.checkpoint.checkpoint(
                    lambda s, rs: upsample_block(
                        hidden_states=s,
                        res_hidden_states_tuple=rs,
                        temb=None,
                    ),
                    sample,
                    res_samples,
                    use_reentrant=False,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    res_hidden_states_tuple=res_samples,
                    temb=None,
                )

        # out
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        sample = self.out_tanh(sample)

        if not return_dict:
            return (sample,)
        return CustomUNet2DOutput(sample=sample)


if __name__ == "__main__":
    model = CustomUNet2D(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        block_out_channels=(64, 128, 256, 512),
        layers_per_block=2,
    )
    x = torch.randn(2, 3, 256, 256)
    y = model(x).sample
    print("plain:", y.shape)  # [2, 3, 256, 256]
