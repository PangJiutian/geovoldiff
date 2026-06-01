"""
Author: Qi Pang 
Description: Conditional 3D Geological Volumes generation pipepine
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
import inspect
from typing import Optional, Union, List

from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from models.unet_condition_3d import MyUNet3DCondition
from models.vae_3d import AutoencoderKL3D
from models.controlnet_3d import MyControlNet3D


class CondGeoVolDiffPipeline(DiffusionPipeline):
    def __init__(self, vae: AutoencoderKL3D, unet: MyUNet3DCondition,
                 scheduler: DDIMScheduler,
                 controlnet: MyControlNet3D):
        super().__init__()
        self.register_modules(vae=vae, unet=unet, scheduler=scheduler, controlnet=controlnet)
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
    
    @torch.no_grad()
    def prepare_controlnet_cond(
        self,
        controlnet_cond: torch.Tensor,
        device,
        dtype,
        do_classifier_free_guidance: bool = False,
    ):
        controlnet_cond = controlnet_cond.to(device=device, dtype=dtype)
        
        if do_classifier_free_guidance:
            null_cond = torch.zeros_like(controlnet_cond)
            controlnet_cond = torch.cat([null_cond, controlnet_cond])
        
        return controlnet_cond
    
    def apply_controlnet(
        self,
        latent_model_input: torch.Tensor,
        t: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ):

        if isinstance(self.controlnet, list):
            down_block_res_samples_list = []
            mid_block_res_sample_list = []
            
            for i, controlnet in enumerate(self.controlnet):
                cond = controlnet_cond[i] if isinstance(controlnet_cond, list) else controlnet_cond
                scale = conditioning_scale[i] if isinstance(conditioning_scale, list) else conditioning_scale
                
                controlnet_output = controlnet(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=cond,
                    conditioning_scale=scale,
                )
                down_block_res_samples_list.append(controlnet_output.down_block_res_samples)
                mid_block_res_sample_list.append(controlnet_output.mid_block_res_sample)
            
            down_block_res_samples = [
                sum(samples) for samples in zip(*down_block_res_samples_list)
            ]
            mid_block_res_sample = sum(mid_block_res_sample_list)
        
        else:
            controlnet_output = self.controlnet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond,
                conditioning_scale=conditioning_scale,
            )
            down_block_res_samples, mid_block_res_sample = (controlnet_output.down_block_res_samples,
                                                            controlnet_output.mid_block_res_sample)
        return down_block_res_samples, mid_block_res_sample
    
    @torch.no_grad()
    def denoise(
        self, 
        latents, 
        num_inference_steps, 
        show_progress: bool = False,
        controlnet_cond: Optional[torch.Tensor] = None,
        conditioning_scale: float = 1.0,
        guidance_scale: float = 1.0,  
        encoder_hidden_states: Optional[torch.Tensor] = None,
        **extra_kwargs
    ):
        self.scheduler.set_timesteps(num_inference_steps, device=latents.device)
        last_timestep = self.scheduler.timesteps[-1]
        timesteps = self.scheduler.timesteps if not show_progress else tqdm(self.scheduler.timesteps, desc="Denoising")
        
        do_classifier_free_guidance = guidance_scale > 1.0
        
        if do_classifier_free_guidance:
            latents = torch.cat([latents] * 2)
            
            if encoder_hidden_states is not None:
                uncond_states = torch.zeros_like(encoder_hidden_states)
                encoder_hidden_states = torch.cat([uncond_states, encoder_hidden_states])
        
        if controlnet_cond is not None and self.controlnet is not None:
            controlnet_cond = self.prepare_controlnet_cond(
                controlnet_cond,
                device=latents.device,
                dtype=latents.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )
        
        for t in timesteps:
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            
            down_block_res_samples = None
            mid_block_res_sample = None
            
            if controlnet_cond is not None and self.controlnet is not None:
                down_block_res_samples, mid_block_res_sample = self.apply_controlnet(
                    latent_model_input=latent_model_input,
                    t=t,
                    controlnet_cond=controlnet_cond,
                    conditioning_scale=conditioning_scale,
                    encoder_hidden_states=encoder_hidden_states,
                )
            
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample
            
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                latents = latents.chunk(2)[1]

            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_kwargs
            ).prev_sample
            
            if do_classifier_free_guidance and t != last_timestep:
                latents = torch.cat([latents] * 2)
        
        return latents

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
        controlnet_cond: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        conditioning_scale: Union[float, List[float]] = 1.0,
        guidance_scale: float = 1.0,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if hasattr(self, "_execution_device"):
            device = self._execution_device
        else:
            device = next(self.unet.parameters()).device
        dtype = self.unet.dtype
        
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
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

        # Denoise with ControlNet
        latents = self.denoise(
            latents,
            num_inference_steps,
            show_progress=show_progress,
            controlnet_cond=controlnet_cond,
            conditioning_scale=conditioning_scale,
            guidance_scale=guidance_scale,
            encoder_hidden_states=encoder_hidden_states,
            **extra_kwargs
        )
        
        # Decode
        images = self.decode_latents(latents)
        return images

