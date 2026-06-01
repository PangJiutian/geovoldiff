# Copyright 2026 Qi Pang. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""
Author: Qi Pang
Description: Sequential 3D axial attention used by the latent diffusion UNet
and ControlNet bottleneck.
"""

import torch.nn.functional as F
from torch import nn
from einops import rearrange


class AxialAttention_2(nn.Module):
    """Axial self-attention over a 3D feature map (B, C, D, H, W).
    """

    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        assert dim % head_dim == 0
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.inner_dim = self.num_heads * head_dim

        self.norm = nn.GroupNorm(32, dim)

        self.to_qkv_d = nn.Conv3d(dim, self.inner_dim * 3, kernel_size=1, bias=False)
        self.to_qkv_h = nn.Conv3d(dim, self.inner_dim * 3, kernel_size=1, bias=False)
        self.to_qkv_w = nn.Conv3d(dim, self.inner_dim * 3, kernel_size=1, bias=False)

        self.to_out = nn.Conv3d(self.inner_dim, dim, kernel_size=1)

    def _axial_attention_d(self, x):
        b, c, d, h, w = x.shape
        qkv = self.to_qkv_d(x)
        q, k, v = qkv.chunk(3, dim=1)

        # (B*H*W, heads, D, head_dim)
        q = rearrange(q, 'b (nh hd) d h w -> (b h w) nh d hd', nh=self.num_heads)
        k = rearrange(k, 'b (nh hd) d h w -> (b h w) nh d hd', nh=self.num_heads)
        v = rearrange(v, 'b (nh hd) d h w -> (b h w) nh d hd', nh=self.num_heads)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, '(b h w) nh d hd -> b (nh hd) d h w', b=b, h=h, w=w)
        return out

    def _axial_attention_h(self, x):
        b, c, d, h, w = x.shape
        qkv = self.to_qkv_h(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (nh hd) d h w -> (b d w) nh h hd', nh=self.num_heads)
        k = rearrange(k, 'b (nh hd) d h w -> (b d w) nh h hd', nh=self.num_heads)
        v = rearrange(v, 'b (nh hd) d h w -> (b d w) nh h hd', nh=self.num_heads)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, '(b d w) nh h hd -> b (nh hd) d h w', b=b, d=d, w=w)
        return out

    def _axial_attention_w(self, x):
        b, c, d, h, w = x.shape
        qkv = self.to_qkv_w(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (nh hd) d h w -> (b d h) nh w hd', nh=self.num_heads)
        k = rearrange(k, 'b (nh hd) d h w -> (b d h) nh w hd', nh=self.num_heads)
        v = rearrange(v, 'b (nh hd) d h w -> (b d h) nh w hd', nh=self.num_heads)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, '(b d h) nh w hd -> b (nh hd) d h w', b=b, d=d, h=h)
        return out

    def forward(self, x, **kwargs):
        """x: (B, C, D, H, W) with (D, H, W) = (depth, inline, xline)."""
        identity = x
        x = self.norm(x)

        x = self._axial_attention_w(x)
        x = self._axial_attention_d(x)
        x = self._axial_attention_h(x)

        x = self.to_out(x)
        return x + identity
