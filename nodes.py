import folder_paths
import os
import comfy.model_management as mm
import torch

from safetensors.torch import load_file
from .upscaler_unet import UpscalerUNet

from diffusers import FlowMatchEulerDiscreteScheduler
from .taef2 import DiffusersTAEF2Wrapper

from .utils import patchify_latents, unpatchify_latents


device = mm.get_torch_device()

folder_paths.folder_names_and_paths["latent_upscale_models"] = (
    [
        os.path.join(folder_paths.models_dir, "latent_upscale_models")
    ],
    {".safetensors"}
)

folder_paths.folder_names_and_paths["vae_approx"] = (
    [
        os.path.join(folder_paths.models_dir, "vae_approx")
    ],
    {".safetensors"}
)



class LoadFlowUpscaler:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list(
                        "latent_upscale_models"
                    ),
                ),
            }
        }

    RETURN_TYPES = ("LATENT_UPSCALER",)
    FUNCTION = "load_model"
    CATEGORY = "latent upscale"

    def load_model(self, model_name):
        flow_upscaler = UpscalerUNet()

        model_path = folder_paths.get_full_path(
            "latent_upscale_models",
            model_name
        )

        state_dict = load_file(model_path)
        flow_upscaler.load_state_dict(state_dict)
        flow_upscaler.eval()

        return (flow_upscaler,)
    

    

class UpscaleLatents:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LATENT_UPSCALER",),
                "latent": ("LATENT",),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"

    def upscale(self, model, latent, seed):

        device = mm.get_torch_device()

        latents_small = latent["samples"]
        latents_small = unpatchify_latents(latents_small)
        latents_small = latents_small.to(device)

        target_latent_height = latents_small.shape[2] * 2

        target_latent_width = latents_small.shape[3] * 2

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        scheduler = FlowMatchEulerDiscreteScheduler()

        scheduler.set_timesteps(1, mu=1.0)
        latents = torch.normal(
            mean=0,
            std=1,
            size=(1, 32, target_latent_height, target_latent_width),
            dtype=latents_small.dtype,
            device=device,
            generator=generator,
        )
        model.eval()

        model = model.to(device)

        for t in scheduler.timesteps:
            latent_model_input = latents
            t = t.to(device).view(1)
            predicted_noise = model(
                sample=latent_model_input,
                timestep=t,
                latents_small=latents_small,
            )
            latents = scheduler.step(predicted_noise, t, latents).prev_sample

        latents = patchify_latents(latents)

        return ({"samples": latents},)
    

class ComfyUITAEF2Wrapper:
    def __init__(self, diffusers_taef2: DiffusersTAEF2Wrapper):
        self.taef2 = diffusers_taef2

    def encode(self, x):
        device = mm.get_torch_device()
        x = x.to(device)
        self.taef2.to(device)
        x = x.permute(0, 3, 1, 2)
        x = x.mul(2).sub(1)
        x = self.taef2.encode(x).latent_dist.sample()
        x = patchify_latents(x)

        return x


    def decode(self, x):
        device = mm.get_torch_device()
        x = x.to(device)
        self.taef2.to(device)
        x = unpatchify_latents(x)
        x = self.taef2.decode(x, return_dict=False)[0]
        x = x.permute(0, 2, 3, 1)
        x = x.add(1).div(2)

        return x


class LoadTAEF2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list(
                        "vae_approx"
                    ),
                ),
            }
        }
    
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_model"
    CATEGORY = "vae"

    def load_model(self, model_name):
        

        model_path = folder_paths.get_full_path(
            "vae_approx",
            model_name
        )

        taef2 = DiffusersTAEF2Wrapper(model_path)
        taef2.eval()

        comfyui_wrapper = ComfyUITAEF2Wrapper(diffusers_taef2=taef2)

        return (comfyui_wrapper,)

NODE_CLASS_MAPPINGS = {
    "LoadFlowUpscaler": LoadFlowUpscaler, "UpscaleLatents": UpscaleLatents, "LoadTAEF2": LoadTAEF2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFlowUpscaler": "Load Flow Upscaler", "UpscaleLatents": "Upscale Latents", "LoadTAEF2": "Load TAEF2"
}