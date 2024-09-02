# ControlNet training guide

## Background

ControlNet models are capable of many tasks, which depend on the conditioning data given at training time.

Example (taken from the Diffusers ControlNet model card):

![example](https://tripleback.net/public/controlnet-example-1.png)

On the left, you can see the "canny edge map" given as the conditioning input. To the right of that are the outputs the ControlNet model guided out of the base SDXL model.

When the model is used in this way, the prompt handles almost none of the composition, merely filling in the details.

## What training a ControlNet looks like

At first, when training a ControlNet, it has zero indication of control:

![example](https://tripleback.net/public/controlnet-example-2.png)
(_ControlNet trained for just 4 steps on a Stable Diffusion 2.1 model_)


The antelope prompt still has a majority of control over the composition, and the ControlNet conditioning input is ignored.

Over time, the control input should be respected:

![example](https://tripleback.net/public/controlnet-example-3.png)
(_ControlNet trained for just 100 steps on a Stable Diffusion XL model_)

At that point, a few indications of ControlNet influence began to appear, but the results were incredibly inconsistent.

A lot more than 100 steps will be needed for this to work!

## Example dataloader configuration

The dataloader configuration remains pretty close to a typical text-to-image dataset configuration:

- The main image data is the `antelope-data` set
  - The key `conditioning_data` is now set, and it should be set to the `id` value of your conditioning data that pairs with this set.
  - `dataset_type` should be `image` for the base set
- A secondary dataset is configured, called `antelope-conditioning`
  - The name isn't important - adding `-data` and `-conditioning` is only done in this example for illustrative purposes.
  - The `dataset_type` should be set to `conditioning`, indicating to the trainer that this is to be used for evaluation and conditioned input training purposes.
- Conditioning inputs are not VAE-encoded, but instead passed into the model directly during training time as pixel values. This means we don't spend any more time processing VAE embeds at the start of training!
- Though everything is explicitly labeled as `-controlnet` here, you can reuse the same text embeds that you used for normal full/LoRA tuning. ControlNet inputs do not modify the prompt embeds.
- The dataset components are labeled as being for SDXL, but they're model-agnostic other than the `resolution` values you'd typically use.
- You likely want cropping enabled with `crop_style='center'` or `crop_style='corner'` so that the perturbations resulting from random crops don't impact your controlnet model training.
  - This is a limitation in the current version of ControlNet training, it can be improved for future releases.

```json
[
    {
        "id": "antelope-data",
        "type": "local",
        "dataset_type": "image",
        "conditioning_data": "antelope-conditioning",
        "instance_data_dir": "/Volumes/ml/datasets/canny-edge/animals/antelope-data",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "an antelope",
        "metadata_backend": "discovery",
        "minimum_image_size": 1.0,
        "maximum_image_size": 1.0,
        "target_downsample_size": 1.0,
        "cache_dir_vae": "/Volumes/ml/cache/vae/sdxl/antelope-data",
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution": 1.0,
        "resolution_type": "area",
        "cache_file_suffix": "controlnet-sdxl"
    },
    {
        "id": "antelope-conditioning",
        "type": "local",
        "dataset_type": "conditioning",
        "instance_data_dir": "/Volumes/ml/datasets/canny-edge/animals/antelope-conditioning",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "an antelope",
        "metadata_backend": "discovery",
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution": 1.0,
        "resolution_type": "area",
        "cache_file_suffix": "controlnet-sdxl"
    },
    {
        "id": "an example backend for text embeds.",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "/Volumes/ml/cache/text/sdxl-base/controlnet"
    }
]
```

## Generating conditioning image inputs

As new as ControlNet support is in SimpleTuner, we've currently just got one option available for generating your training set:

- [create_canny_edge.py](/toolkit/datasets/controlnet/create_canny_edge.py)
  - An extremely basic example on generating a training set for Canny model training.
  - You will have to modify the `input_dir` and `output_dir` values in the script

This will take about 30 seconds for a small dataset of fewer than 100 images.

## Modifying your environment file to train ControlNet models

Just setting up the dataloader configuration won't be enough to start training ControlNet models.

Inside `config/config.json`, you will have to set the following values:

```bash
"model_type": 'full',
"controlnet": true,

# You may have to reduce TRAIN_BATCH_SIZE more than usual
"train_batch_size": 1
```

## Inference on resulting ControlNet models

An SDXL example is provided here:

```py
# Update these values:
base_model = "stabilityai/stable-diffusion-xl-base-1.0"         # This is the model you used as `--pretrained_model_name_or_path`
controlnet_model_path = "diffusers/controlnet-canny-sdxl-1.0"   # This is the path to the resulting ControlNet checkpoint
# controlnet_model_path = "/path/to/controlnet/checkpoint-100"

# Leave the rest alone:
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import torch
import numpy as np
import cv2

prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
negative_prompt = 'low quality, bad quality, sketches'

image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png")

controlnet_conditioning_scale = 0.5  # recommended for good generalization

controlnet = ControlNetModel.from_pretrained(
    controlnet_model_path,
    torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()

image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = Image.fromarray(image)

images = pipe(
    prompt, negative_prompt=negative_prompt, image=image, controlnet_conditioning_scale=controlnet_conditioning_scale,
    ).images

images[0].save(f"hug_lab.png")
```
(_Demo code lifted from the [Hugging Face SDXL ControlNet example](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0)_)