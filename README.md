# Flow Upscaler ComfyUI Nodes

**Flow Upscaler** is a fast latent upscaler model that works in the [Flux.2](https://bfl.ai/models/flux-2) latent space.

Under the hood, it is a lightweight **Rectified Flow** model with **59M** parameters that generates upscaled latents in a single denoising step.

**[Download Weights](https://huggingface.co/TensorForger/FlowUpscaler)**

Features:

* Upscaling from **512x512** to **1024x1024** takes **8ms*** 
* The model is trained for **2X** upscaling, but multiple passes can be chained to reach up to **8K** resolution
* A full pipeline with Flux generation, upscaling to **8K**, and decoding runs in just **25 seconds** (on RTX 5090)
* The training process uses **Flow Distillation** with Flux.2 as a teacher, forcing the model to learn strong image semantics

*On RTX 5090, in latent space, without decoding, see benchmark [here](https://github.com/tensorforger/CTGMWorkshop).

![comparison](https://raw.githubusercontent.com/tensorforger/tensorforger/main/assets/upscaler_comparison.png)

Here is one **4X** upscaled image (workflow included):

![example](https://raw.githubusercontent.com/tensorforger/tensorforger/main/assets/upscaled_cat.png)

## Quick start

1. Clone this repository into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/TensorForger/comfyui-flow-upscaler
````

2. Download the model [here](https://huggingface.co/TensorForger/FlowUpscaler) and place it in:

```
ComfyUI/models/latent_upscale_models
```

3. Optional, but recommended: Download the tiny Flux.2 VAE from [here](https://huggingface.co/madebyollin/taef2) and place it in:

```
ComfyUI/models/vae_approx
```

Flow Upscaler can generate very large images, and decoding them with the original Flux.2 VAE becomes impractical. It is recommended to use the tiny VAE for resolutions of **2048x2048** and above.

4. Open ComfyUI and load the demo workflow (`workflow.json` in the root of this repository).

## How it works

Architecturally, Flow Upscaler is a U-Net with SDXL-style ResNet blocks. It takes a noisy sample as input and predicts velocity as output. The generation process happens directly in high-resolution latent space.

The low-resolution latents are passed through a separate conditioning encoder that produces control signals, which are injected into the main U-Net encoder using FiLM conditioning.

No attention layers are used, so compute scales linearly with image area. This makes generation at **8K** resolution possible.

![example](https://raw.githubusercontent.com/tensorforger/tensorforger/main/assets/flow_upscaler_architecture.PNG)

The model is trained using **Flow Distillation** with Flux.2-klein-4B as a teacher. We generated **20K** diverse images with Flux, storing the initial noise, generated latents, and downscaled latents used for conditioning.

The downscaled latents are created by decoding high-resolution latents, downscaling them in pixel space, and encoding them back into latents. Direct latent downscaling introduces artifacts and breaks latent patterns, resulting in blurry decoded images.

![example](https://raw.githubusercontent.com/tensorforger/tensorforger/main/assets/flow_upscaler_training_approach.PNG)

## Training code

If you want to explore the training code or use the model outside ComfyUI, see:

`notebooks/flow_upscaler` in [https://github.com/tensorforger/CTGMWorkshop](https://github.com/tensorforger/CTGMWorkshop)