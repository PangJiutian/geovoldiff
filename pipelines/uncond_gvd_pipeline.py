"""
Author: Qi Pang 
Description:  Unconditional 3D Geological Volumes generation pipepine
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
import inspect

from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from models.unet_condition_3d import MyUNet3DCondition
from models.vae_3d import AutoencoderKL3D


class GeoVolDiffPipeline(DiffusionPipeline):
    def __init__(self, vae: AutoencoderKL3D, unet: MyUNet3DCondition,
                 scheduler: DDIMScheduler):
        super().__init__()
        self.register_modules(vae=vae, unet=unet, scheduler=scheduler)
        self.use_vae = vae is not None

    @torch.no_grad()
    def prepare_latents(
            self,
            T: int,
            batch_size: int,
            num_channels: int,
            height: int,
            width: int,
            dtype,
            device,
            generator
    ):

        if self.use_vae:
            shape = (batch_size, num_channels, T, height, width)
        else:
            shape = (batch_size, 1, T, height, width)

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Denoising Loop
    @torch.no_grad()
    def denoise(self, latents, num_inference_steps, show_progress: bool = False, **extra_kwargs):
        # Do not forget 'latents' must in the loop (iterative updated)!!!!
        self.scheduler.set_timesteps(num_inference_steps, device=latents.device)

        timesteps = self.scheduler.timesteps if not show_progress else tqdm(self.scheduler.timesteps, desc="Denoising")
        for t in timesteps:
            latents = self.scheduler.scale_model_input(latents, t)
            noise_pred = self.unet(latents, t).sample
            latents = self.scheduler.step(noise_pred, t, latents, **extra_kwargs).prev_sample
        return latents

    # Decode
    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor):
        if self.use_vae:
            latents = latents / self.vae.config.scaling_factor
            images = self.vae.decode(latents).sample
        else:
            images = latents
        return images

    @torch.no_grad()
    def encode(self, images: torch.Tensor):
        if self.use_vae:
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        else:
            latents = images

        return latents

    @torch.no_grad()
    def __call__(
            self,
            T: int = 32,
            H: int = 32,
            W: int = 32,
            batch_size: int = 1,
            num_inference_steps: int = 50,
            eta: float = 0.0,
            generator=None,
            show_progress: bool = False,
            **kwargs,
    ):
        if hasattr(self, "_execution_device"):
            device = self._execution_device
        else:
            device = next(self.unet.parameters()).device
        dtype = self.unet.dtype
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        extra_kwargs = {"generator": generator}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        # Prepare latents
        latents = self.prepare_latents(
            T=T,
            batch_size=batch_size,
            num_channels=self.unet.config.in_channels,
            height=H,
            width=W,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        # Denoise
        latents = self.denoise(
            latents,
            num_inference_steps,
            show_progress=show_progress,
            **extra_kwargs
        )

        # Decode
        images = self.decode_latents(latents)
        return images

