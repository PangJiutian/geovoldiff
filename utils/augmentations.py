"""
Author: Qi Pang 
Description: Augmentation
"""
import torch
import torch.nn.functional as F
from typing import Tuple
import random

class SeismicAugmentation3D:
    def __init__(
            self,
            # Geometric (rigid only)
            flip_prob: float = 0.5,
            rotate90_prob: float = 0.3,

            # Intensity
            intensity_shift_prob: float = 0.5,
            intensity_shift_range: Tuple[float, float] = (-0.05, 0.05),
            intensity_scale_prob: float = 0.5,
            intensity_scale_range: Tuple[float, float] = (0.9, 1.1),

            # Noise
            noise_prob: float = 0.3,
            noise_std_range: Tuple[float, float] = (0.005, 0.02),
    ):
        self.flip_prob = flip_prob
        self.rotate90_prob = rotate90_prob

        self.intensity_shift_prob = intensity_shift_prob
        self.intensity_shift_range = intensity_shift_range

        self.intensity_scale_prob = intensity_scale_prob
        self.intensity_scale_range = intensity_scale_range

        self.noise_prob = noise_prob
        self.noise_std_range = noise_std_range

    def __call__(self, volume: torch.Tensor, is_label: bool = False) -> torch.Tensor:
        squeeze_output = False
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)
            squeeze_output = True

        volume = self._rigid_geometric(volume)
        
        if not is_label:
            volume = self._intensity_transform(volume)
    
            if torch.rand(1).item() < self.noise_prob:
                volume = self._add_noise(volume)

        if squeeze_output:
            volume = volume.squeeze(0)

        return volume

    def _rigid_geometric(self, volume: torch.Tensor) -> torch.Tensor:
        # Flip along spatial axes
        if torch.rand(1).item() < self.flip_prob:
            volume = torch.flip(volume, dims=[1])  

        if torch.rand(1).item() < self.flip_prob:
            volume = torch.flip(volume, dims=[2])  

        # Rotate in H-W plane only
        if torch.rand(1).item() < self.rotate90_prob:
            k = torch.randint(1, 4, (1,)).item()
            volume = torch.rot90(volume, k, dims=[1, 2])

        return volume

    def _intensity_transform(self, volume: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.intensity_shift_prob:
            shift = torch.empty(1).uniform_(*self.intensity_shift_range).item()
            volume = volume + shift

        if torch.rand(1).item() < self.intensity_scale_prob:
            scale = torch.empty(1).uniform_(*self.intensity_scale_range).item()
            volume = volume * scale

        # Keep normalized range
        volume = torch.clamp(volume, -1.0, 1.0)
        return volume

    def _add_noise(self, volume: torch.Tensor) -> torch.Tensor:
        std = torch.empty(1).uniform_(*self.noise_std_range).item()
        noise = torch.randn_like(volume) * std
        volume = volume + noise
        volume = torch.clamp(volume, -1.0, 1.0)
        return volume
    
    def _frequency_mask(self, volume: torch.Tensor) -> torch.Tensor:
        # FFT along spatial dimensions
        fft_vol = torch.fft.fftn(volume, dim=(-3, -2, -1))

        # Random mask ratio
        mask_ratio = random.uniform(*self.freq_mask_ratio_range)

        # Create random mask
        C, D, H, W = volume.shape
        mask = torch.rand(C, D, H, W, device=volume.device) > mask_ratio

        # Apply mask
        fft_vol = fft_vol * mask

        # Inverse FFT
        volume = torch.fft.ifftn(fft_vol, dim=(-3, -2, -1)).real
        volume = torch.clamp(volume, -1.0, 1.0)

        return volume

    def _local_contrast(self, volume: torch.Tensor) -> torch.Tensor:
        contrast_factor = random.uniform(*self.contrast_range)

        # Compute local mean
        kernel_size = 7
        local_mean = F.avg_pool3d(
            volume,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )

        # Enhance contrast around local mean
        volume = local_mean + (volume - local_mean) * contrast_factor
        volume = torch.clamp(volume, -1.0, 1.0)

        return volume