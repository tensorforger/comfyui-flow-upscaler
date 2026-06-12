# Flow Upscaler ComfyUI nodes

**Flow Upscaler** is a fast Latent Upscaler model that works in [Flux.2](https://bfl.ai/models/flux-2) latent space.

Under the hood, it is a lightweight **Rectified flow** model with **59M** parameters generating upscaled latents in just one denoising step.

**[Download Weights](https://huggingface.co/TensorForger/FlowUpscaler)**

Features:

* Upscaling latents for image from **512x512** to **1024x1024** on RTX 5090 takes **7ms**
* The model is trained only for **2X** upscaling, but you can chain it many times up to **8K** resolution
* The training process involves **Flow Distillation** with Flux.2 as a teacher what forces it to understand image semantic very well

Here is one **4X** upscaled image (two passes):
![example](https://raw.githubusercontent.com/tensorforger/tensorforger/main/assets/upscaled_cat.png)

## Quick start

1. Clone this repository to `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/TensorForger/comfyui-flow-upscaler
```

2. Download the model [here](https://huggingface.co/TensorForger/FlowUpscaler) and place it under `ComfyUI/models/latent_upscale_models`

3. Optional, but recommended: Download tiny Flux.2 VAE from [here](https://huggingface.co/madebyollin/taef2) and place it under `ComfyUI/models/vae_approx`.
Flow Upscaler allows to generate really large images and decoding them with original Flux.2 VAE becomes impossible. It is recommended to use tiny VAE
starting from resolution 2048x2048 and larger.

4. Open ComfyUI and drop demo workflow (file `workflow.json` in the root of this repo)


## How it works

Architecturally, Flow Upscaler is a Unet with SDXL-style ResNet blocks. It takes the noisy sample on input and predicts velocity on output. This generation process happens in high resolution space. The low resolution latents are passed in a separate conditioning encoder that emits control signals that are passed to main Unet encoder through FiLM conditioning.

No attention is used, so compute scales linearly with image area. This makes generation in 8K possible.

![example](https://raw.githubusercontent.com/tensorforger/tensorforger/main/assets/flow_upscaler_architecture.PNG)

The model is trained through Flow Distillation with Flux.2-klein-4B as a teacher. We generated 20K various images with Flux storing initial noise, generated latents and downscaled latents for conditioning. The downscaled latents are generated throgh decoding high resolution latents, downscaling in pixel space and encoding back to latents because downscaling directly in latents breaks some "latent patterns" that makes image blurry if you decode it.

![example](https://raw.githubusercontent.com/tensorforger/tensorforger/main/assets/flow_upscaler_training_approach.PNG)


## Training code

If you want to explore training code or use model outside of ComfyUI directly from code, see `notebooks/flow_upscaler` in [https://github.com/tensorforger/CTGMWorkshop](https://github.com/tensorforger/CTGMWorkshop)