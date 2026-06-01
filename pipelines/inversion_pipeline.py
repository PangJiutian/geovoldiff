import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Union

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput

from models.unet_custom_2d import CustomUNet2D

Array = Union[np.ndarray, torch.Tensor]


def pad_to_multiple(x: Array, multiple: int = 8) -> Tuple[Array, Tuple[int, int]]:
    """Right/bottom reflect-pad the last two dims so each is divisible by `multiple`.

    Args:
        x: array or tensor with at least 2 trailing spatial dims (..., H, W).
        multiple: the divisor (typically 2**num_downsamples).

    Returns:
        padded: padded array/tensor.
        (pad_h, pad_w): the padding amounts applied to H and W.
    """
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if isinstance(x, np.ndarray):
        pad_width = [(0, 0)] * (x.ndim - 2) + [(0, pad_h), (0, pad_w)]
        x_padded = np.pad(x, pad_width, mode="reflect")
    else:
        x_padded = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    return x_padded, (pad_h, pad_w)


def unpad(x: Array, pad_h: int, pad_w: int) -> Array:
    """Strip the padding added by :func:`pad_to_multiple` from the last two dims."""
    if pad_h > 0 and pad_w > 0:
        return x[..., :-pad_h, :-pad_w]
    if pad_h > 0:
        return x[..., :-pad_h, :]
    if pad_w > 0:
        return x[..., :, :-pad_w]
    return x


@dataclass
class SeismicInversionPipelineOutput(BaseOutput):
    """Output of :class:`SeismicInversionPipeline`.

    Attributes:
        impedance: Predicted impedance volume in log-normalized model space,
            shape ``(B, 1, H, W)``. Apply :func:`geovoldiff.anti_normalize` with
            the appropriate ``mean`` / ``std`` to recover physical units.
    """
    impedance: torch.Tensor

class SeismicInversionPipeline(DiffusionPipeline):
    """Single-step regression pipeline: normalized seismic -> log-impedance."""

    def __init__(self, unet: CustomUNet2D):
        super().__init__()
        self.register_modules(unet=unet)

    @torch.no_grad()
    def __call__(
        self,
        seismic: torch.Tensor,
        pad_multiple: int = 8,
        return_dict: bool = True,
    ):
        """
        Args:
            seismic: normalized seismic tensor, shape ``(B, 1, H, W)``.
            pad_multiple: spatial divisor for reflect padding. Must be a
                multiple of ``2 ** num_downsamples`` of the UNet.
            return_dict: if False, return a tuple ``(impedance,)``.
        """
        device = self._execution_device if hasattr(self, "_execution_device") else next(self.unet.parameters()).device
        seismic = seismic.to(device=device, dtype=self.unet.dtype)

        padded, (pad_h, pad_w) = pad_to_multiple(seismic, multiple=pad_multiple)
        impedance = self.unet(padded).sample
        impedance = unpad(impedance, pad_h, pad_w)

        if not return_dict:
            return (impedance,)
        return SeismicInversionPipelineOutput(impedance=impedance)