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
# Description: Extended AutoencoderKL to 3D architecture for
#   volumetric seismic data processing.
#   Added TemporalAttention: original parallel axial attention
#   module for memory-efficient 3D bottleneck attention.
# Original source: https://github.com/huggingface/diffusers
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from einops import rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.activations import GEGLU, GELU

@dataclass
class AutoencoderKLOutput:
    """Output of AutoencoderKL encoding method."""
    latent_dist: 'DiagonalGaussianDistribution'


@dataclass
class DecoderOutput:
    """Output of decoding method."""
    sample: torch.Tensor


class DiagonalGaussianDistribution:
    """
    Gaussian Distribution with diagonal covariance matrix.
    """

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self) -> torch.Tensor:
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self, other=None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3, 4]
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3, 4]
                )

class ResnetBlock3D(nn.Module):
    """
    3D Residual Block
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            dropout: float = 0.0,
            groups: int = 32,
            eps: float = 1e-6,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        self.nonlinearity = nn.SiLU()

        self.conv_shortcut = None
        if self.in_channels != self.out_channels:
            self.conv_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x

        h = self.norm1(h)
        h = self.nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)

        return x + h

class TemporalAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            head_dim: int,
            num_heads: int
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.inner_dim = num_heads * head_dim

        self.to_qkv_depth = nn.Conv3d(dim, self.inner_dim * 3, kernel_size=1)
        self.to_qkv_height = nn.Conv3d(dim, self.inner_dim * 3, kernel_size=1)
        self.to_qkv_width = nn.Conv3d(dim, self.inner_dim * 3, kernel_size=1)

        self.to_out = nn.Conv3d(self.inner_dim, dim, kernel_size=1)

        self._gradient_checkpointing = False

    def _axial_attention(self, x, qkv_layer, axis):

        b, c, d, h, w = x.shape

        qkv = qkv_layer(x)
        q, k, v = qkv.chunk(3, dim=1)

        if axis == 2:  # depth axis
            # (B, C, D, H, W) -> (B, H, W, C, D)
            q = q.permute(0, 3, 4, 1, 2).contiguous()
            k = k.permute(0, 3, 4, 1, 2).contiguous()
            v = v.permute(0, 3, 4, 1, 2).contiguous()
            batch_size, spatial_h, spatial_w = b, h, w
            seq_len = d
        elif axis == 3:  # height axis
            # (B, C, D, H, W) -> (B, D, W, C, H)
            q = q.permute(0, 2, 4, 1, 3).contiguous()
            k = k.permute(0, 2, 4, 1, 3).contiguous()
            v = v.permute(0, 2, 4, 1, 3).contiguous()
            batch_size, spatial_h, spatial_w = b, d, w
            seq_len = h
        else:  # width axis
            # (B, C, D, H, W) -> (B, D, H, C, W)
            q = q.permute(0, 2, 3, 1, 4).contiguous()
            k = k.permute(0, 2, 3, 1, 4).contiguous()
            v = v.permute(0, 2, 3, 1, 4).contiguous()
            batch_size, spatial_h, spatial_w = b, d, h
            seq_len = w

        q = q.view(batch_size * spatial_h * spatial_w, self.num_heads, self.head_dim, seq_len)
        k = k.view(batch_size * spatial_h * spatial_w, self.num_heads, self.head_dim, seq_len)
        v = v.view(batch_size * spatial_h * spatial_w, self.num_heads, self.head_dim, seq_len)

        q = q.transpose(2, 3)  # (B*H*W, heads, seq_len, head_dim)
        k = k.transpose(2, 3)
        v = v.transpose(2, 3)

        out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(2, 3).contiguous()
        out = out.view(batch_size, spatial_h, spatial_w, c, seq_len)

        if axis == 2:
            out = out.permute(0, 3, 4, 1, 2).contiguous()  # -> (B, C, D, H, W)
        elif axis == 3:
            out = out.permute(0, 3, 4, 1, 2).contiguous()  # -> (B, C, H, D, W) then swap
            out = out.permute(0, 1, 3, 2, 4).contiguous()  # -> (B, C, D, H, W)
        else:
            out = out.permute(0, 3, 1, 2, 4).contiguous()  # -> (B, C, D, H, W)

        return out

    def _forward(self, hidden_states):
        b, l, c = hidden_states.shape
        d = h = w = int(round(l ** (1 / 3)))
        hidden_states = rearrange(hidden_states, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)

        out_d = self._axial_attention(hidden_states, self.to_qkv_depth, axis=2)
        out_h = self._axial_attention(hidden_states, self.to_qkv_height, axis=3)
        out_w = self._axial_attention(hidden_states, self.to_qkv_width, axis=4)

        hidden_states = (out_d + out_h + out_w) / 3.0
        hidden_states = self.to_out(hidden_states)
        hidden_states = rearrange(
            hidden_states, "b c d h w -> b (d h w) c")
        return hidden_states

    def forward(self, hidden_states):
        if self.training and getattr(self, '_gradient_checkpointing', False):
            return torch.utils.checkpoint.checkpoint(self._forward, hidden_states, use_reentrant=False)
        return self._forward(hidden_states)


class FeedForward(nn.Module):
    """
    Standard FeedForward
    """

    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            mult: int = 4,
            dropout: float = 0.0,
            activation_fn: str = "geglu",
            final_dropout: bool = False,
            inner_dim: Optional[int] = None,
            bias: bool = True,
    ):
        super().__init__()

        if inner_dim is None:
            inner_dim = int(dim * mult)

        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        elif activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = GEGLU(dim, inner_dim, approximate="tanh", bias=bias)
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

        self.net = nn.ModuleList([
            act_fn,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(inner_dim, dim_out, bias=bias),
            nn.Dropout(dropout) if final_dropout else nn.Identity(),
        ])

    def forward(self, hidden_states, *args, **kwargs):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class MyTransformerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            attention_head_dim: int,
            num_attention_heads: int,
            dropout=0.0,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            norm_elementwise_affine: bool = True,
            # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
            norm_type: str = "layer_norm",
            norm_eps: float = 1e-5,
            final_dropout: bool = False,
            ff_inner_dim: Optional[int] = None,
            ff_bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.inner_dim = self.num_attention_heads * self.attention_head_dim

        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_zero = norm_type == "ada_norm_zero"
        self.use_ada_layer_norm_continuous = norm_type == "ada_norm_continuous"
        self.use_layer_norm = norm_type == "layer_norm"

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = TemporalAttention(
            dim=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim, )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.ff = FeedForward(
            dim=dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        b, c, d, h, w = hidden_states.shape
        hidden_states = rearrange(hidden_states, "b c d h w -> b (d h w) c")

        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states)
        hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm2(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states
        
        return rearrange(hidden_states, "b (d h w) c -> b c d h w", d=d, h=h, w=w)


class Downsample3D(nn.Module):
    """3D Downsampling layer"""

    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv

        if use_conv:
            self.conv = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.conv = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    """3D Upsampling layer"""

    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv

        if use_conv:
            self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x


class Encoder3D(nn.Module):
    """
    3D Encoder
    """

    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 4,
            block_out_channels: Tuple[int] = (128, 256, 512, 512),
            layers_per_block: int = 2,
            norm_num_groups: int = 32,
            double_z: bool = True,
            mid_block_add_attention: bool = True,
            attention_head_dim: int = 8,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        # Initial convolution
        self.conv_in = nn.Conv3d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        # Downsampling blocks
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]

        for i, out_ch in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = out_ch
            is_final_block = i == len(block_out_channels) - 1

            down_block = nn.ModuleList([])

            for j in range(self.layers_per_block):
                in_ch = input_channel if j == 0 else output_channel
                down_block.append(ResnetBlock3D(in_ch, output_channel, groups=norm_num_groups))

            if not is_final_block:
                down_block.append(Downsample3D(output_channel))

            self.down_blocks.append(down_block)

        # Middle block
        self.mid_block = nn.ModuleList([
            ResnetBlock3D(block_out_channels[-1], block_out_channels[-1], groups=norm_num_groups),
        ])

        if mid_block_add_attention:
            self.mid_block.append(MyTransformerBlock(block_out_channels[-1], 
                                                     attention_head_dim,
                                                     block_out_channels[-1] // attention_head_dim))

        self.mid_block.append(
            ResnetBlock3D(block_out_channels[-1], block_out_channels[-1], groups=norm_num_groups)
        )

        # Output
        self.conv_norm_out = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=block_out_channels[-1])
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv3d(block_out_channels[-1], conv_out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv
        h = self.conv_in(x)

        # Downsampling
        for down_block in self.down_blocks:
            for layer in down_block:
                h = layer(h)
                # print(h.shape,'down')
        # Middle
        for layer in self.mid_block:
            h = layer(h)
            # print(h.shape,'mid')

        # Output
        h = self.conv_norm_out(h)
        h = self.conv_act(h)
        h = self.conv_out(h)

        return h


class Decoder3D(nn.Module):
    """
    3D Decoder
    """

    def __init__(
            self,
            in_channels: int = 4,
            out_channels: int = 1,
            block_out_channels: Tuple[int] = (128, 256, 512, 512),
            layers_per_block: int = 2,
            norm_num_groups: int = 32,
            mid_block_add_attention: bool = True,
            attention_head_dim: int = 8,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        # Initial convolution
        self.conv_in = nn.Conv3d(in_channels, block_out_channels[-1], kernel_size=3, padding=1)

        # Middle block
        self.mid_block = nn.ModuleList([
            ResnetBlock3D(block_out_channels[-1], block_out_channels[-1], groups=norm_num_groups),
        ])

        self.mid_block.append(MyTransformerBlock(block_out_channels[-1], 
                                                 attention_head_dim,
                                                 block_out_channels[-1] // attention_head_dim))

        self.mid_block.append(
            ResnetBlock3D(block_out_channels[-1], block_out_channels[-1], groups=norm_num_groups)
        )

        # Upsampling blocks
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]

        for i, out_ch in enumerate(reversed_block_out_channels):
            input_channel = output_channel
            output_channel = out_ch
            is_final_block = i == len(block_out_channels) - 1

            up_block = nn.ModuleList([])

            for j in range(self.layers_per_block + 1):
                in_ch = input_channel if j == 0 else output_channel
                up_block.append(ResnetBlock3D(in_ch, output_channel, groups=norm_num_groups))

            if not is_final_block:
                up_block.append(Upsample3D(output_channel))

            self.up_blocks.append(up_block)

        # Output
        self.conv_norm_out = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=block_out_channels[0])
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv3d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Initial conv
        h = self.conv_in(z)

        # Middle
        for layer in self.mid_block:
            h = layer(h)

        # Upsampling
        for up_block in self.up_blocks:
            for layer in up_block:
                h = layer(h)

        # Output
        h = self.conv_norm_out(h)
        h = self.conv_act(h)
        h = self.conv_out(h)

        return h


class AutoencoderKL3D(ModelMixin, ConfigMixin):
    """
    3D Variational Autoencoder (VAE) with KL divergence loss.
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale seismic data)
        out_channels: Number of output channels (default: 1)
        latent_channels: Number of latent channels (default: 4)
        block_out_channels: Channel multipliers for each resolution level
        layers_per_block: Number of ResNet blocks per resolution level
        norm_num_groups: Number of groups for GroupNorm
        scaling_factor: Scaling factor for latent space (default: 0.18215, same as SD)
        mid_block_add_attention: Whether to add attention in middle block
    
    Example:
        >>> vae = AutoencoderKL3D(
        ...     in_channels=1,
        ...     out_channels=1,
        ...     latent_channels=4,
        ...     block_out_channels=(64, 128, 256, 512),
        ... )
        >>> 
        >>> # Encoding
        >>> x = torch.randn(2, 1, 128, 256, 256)  # (B, C, D, H, W)
        >>> posterior = vae.encode(x)
        >>> z = posterior.latent_dist.sample()  # (B, 4, 16, 32, 32) if 8x downsampling
        >>> 
        >>> # Decoding
        >>> x_recon = vae.decode(z)  # (B, 1, 128, 256, 256)
    """

    @register_to_config
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            latent_channels: int = 4,
            block_out_channels: Tuple[int] = (128, 256, 512, 512),
            layers_per_block: int = 2,
            norm_num_groups: int = 32,
            scaling_factor: float = 0.18215,
            mid_block_add_attention: bool = True,
            attention_head_dim: int = 8,
    ):
        super().__init__()

        self.encoder = Encoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            double_z=True,
            mid_block_add_attention=mid_block_add_attention,
            attention_head_dim=attention_head_dim,
        )

        self.decoder = Decoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            mid_block_add_attention=mid_block_add_attention,
            attention_head_dim=attention_head_dim
        )

        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)

        self.latent_channels = latent_channels
        self.scaling_factor = scaling_factor

    def encode(self, x: torch.Tensor, return_dict: bool = True) -> Union[AutoencoderKLOutput, Tuple]:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            return_dict: Whether to return AutoencoderKLOutput
            
        Returns:
            AutoencoderKLOutput with latent_dist attribute
        """
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        """
        Decode latent representation to output.
        
        Args:
            z: Latent tensor of shape (B, latent_channels, D', H', W')
            return_dict: Whether to return DecoderOutput
            
        Returns:
            Decoded tensor of shape (B, out_channels, D, H, W)
        """
        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
            self,
            sample: torch.Tensor,
            sample_posterior: bool = False,
            return_dict: bool = True,
    ) -> Union[DecoderOutput, Tuple]:
        """
        Forward pass through the autoencoder.
        
        Args:
            sample: Input tensor (B, C, D, H, W)
            sample_posterior: Whether to sample from posterior (True) or use mode (False)
            return_dict: Whether to return DecoderOutput
            
        Returns:
            Reconstructed sample
        """
        posterior = self.encode(sample).latent_dist

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)


if __name__ == "__main__":
    b, c, t, h, w = 1, 1, 32, 32, 32
    sample = torch.randn(b, c, t, h, w)
    vae = AutoencoderKL3D(
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        block_out_channels=(32, 64, 64),
        scaling_factor=1.0,
        mid_block_add_attention=True,
        attention_head_dim=8,
    )
    out = vae.encode(sample).latent_dist
    latent = out.sample()
    x_recon = vae.decode(latent).sample

