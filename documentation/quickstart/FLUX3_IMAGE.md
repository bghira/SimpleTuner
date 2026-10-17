# FLUX.3 Image Klein quickstart

This guide walks you through training a FLUX.3 Image Klein LoRA on your own images. A LoRA is a small adapter that teaches the model a subject or style while keeping the original model weights separate.

Start with `model_family: "flux3_image"`, `model_flavour: "klein-base"`, and `model_type: "lora"`. This selects [black-forest-labs/flux-3-image-klein-base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base). Visit the model page, accept its access conditions, and read the [FLUX [Kom]munity license](https://huggingface.co/black-forest-labs/flux-3-image-klein-base/blob/main/LICENSE) before downloading.

## Installation

You will need Python 3.13, Git, and enough free disk space for the model, your images, and the training cache. The following commands install SimpleTuner for an NVIDIA CUDA GPU. See the [installation guide](../INSTALL.md) for other platforms and CUDA 13 installation instructions.

```bash
git clone https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[cuda]'
hf auth login
```

Run the remaining commands from the `SimpleTuner` directory with the virtual environment active. When `hf auth login` asks for a token, use a Hugging Face token with read access to the model.

## Choose a recipe for your GPU

GPU memory (VRAM) determines how many images you can process together. Choose the example matching your GPU memory. The 24 and 32 GB Base recipes use 512-resolution datasets; the larger recipes also include 1024-resolution datasets.

| GPU memory | Base example | Images per batch at 512 / 1024 |
| --- | --- | --- |
| 24 GB | `flux3_image-24g.peft-lora` | 3 / — |
| 32 GB | `flux3_image-32g.peft-lora` | 4 / — |
| 48 GB | `flux3_image-48g.peft-lora` | 9 / 2 |
| 80 GB | `flux3_image-80g.peft-lora` | 18 / 5 |
| 144 GB | `flux3_image-144g.peft-lora` | 36 / 10 |

For the distilled `klein` flavour, choose one of these recipes. Use its full example name in the configuration-copy commands below. The training scale is a square-equivalent area; image aspect ratios are preserved.

| GPU memory | Distilled example | Training scale | Images per batch |
| --- | --- | --- | ---: |
| 24 GB | `flux3_image-klein-24g.peft-lora` | 768px | 1 |
| 80 GB | `flux3_image-klein-80g.peft-lora` | 1024px | 4 |

The batch sizes are starting points. Longer captions and reference images increase memory use; lower `train_batch_size` if you run out of memory. The recipes use rank-32 LoRA, learning rate `1e-4`, and REPA, which compares image features during training. Their supplied dataloaders include concept images and regularisation images to help retain the model’s broader image knowledge.

## Prepare your images and captions

Create `datasets/my-concept` inside the SimpleTuner directory and place your training images there. Give each image a text file with the same name containing its caption:

```text
datasets/my-concept/
  image01.png
  image01.txt
  image02.jpg
  image02.txt
```

For a subject LoRA, choose a distinctive word such as `TOK` and use it consistently in the captions. Describe what is visible, including the subject, setting, and pose. For example, `image01.txt` might contain:

```text
A photo of TOK, a small ceramic owl with blue feathers, on a wooden table.
```

## Set up the training configuration

Copy the recipe you selected. These commands show `flux3_image-48g.peft-lora`; replace that full example name with your chosen Base or distilled recipe:

```bash
mkdir -p config/examples
cp -R simpletuner/examples/flux3_image-48g.peft-lora config/examples/
cp simpletuner/examples/flux3_image-prompts.json config/examples/
cp config/examples/flux3_image-48g.peft-lora/config.json config/config.json
```

Open `config/config.json` in a text editor. For your first run with the local images above, change the following values inside the existing configuration:

```json
{
  "data_backend_config": "config/dataloader.json",
  "output_dir": "output/my-flux3-lora",
  "train_batch_size": 1
}
```

A batch size of 1 lets you start with a small dataset. Increase it once the first run is working and you have enough images and free GPU memory. Set `max_train_steps` to control the training length. Edit `config/examples/flux3_image-prompts.json` to include prompts with your subject word; these prompts produce progress images during training.

Save the following as `config/dataloader.json`. This file tells SimpleTuner where to find the images, how to read their captions, and where to store cached data:

```json
[
  {
    "id": "my-concept",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "datasets/my-concept",
    "caption_strategy": "textfile",
    "resolution": 0.589824,
    "resolution_type": "area",
    "crop": false,
    "cache_dir_vae": "cache/vae/flux3-image/my-concept",
    "text_embeds": "text"
  },
  {
    "id": "text",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux3-image/my-concept"
  }
]
```

For this local dataset, `resolution: 0.589824` with `resolution_type: "area"` gives a 768×768 equivalent area. Use `1.048576` for the 1024-scale dataset in the 80 GB distilled recipe.

The `resolution` setting controls the training image size; image dimensions are aligned to multiples of 16. The cache stores reusable image latents and text embeddings, so later runs can start faster. Use separate cache directories for different model families. See the [dataloader guide](../DATALOADER.md) to add more datasets or regularisation images.

## Optional: expand captions with the prompt upsampler

Add this setting to `config/config.json` to expand short training captions and validation prompts with the [FLUX.3 Image Klein prompt upsampler](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup):

```json
{
  "flux3_prompt_upsample": true
}
```

Upsampling is off by default. When enabled, SimpleTuner downloads the separate Qwen3.5-9B model (about 19 GB of weights) and rewrites positive prompts before caching their text embeddings. It uses your paired reference images for edit requests. Negative prompts and empty dropout captions are kept unchanged. Your original caption files are preserved.

Allow extra system RAM and preprocessing time. The upsampler and text encoder take turns on the GPU, and the resulting embeddings are reused during training. SimpleTuner keeps expanded and original prompt embeddings separate, and reuses cached results when the same references are selected again.

## Start training

From the SimpleTuner directory, run:

```bash
simpletuner train
```

On the first run, SimpleTuner downloads the model, reads your images, and creates the text and image caches before training begins. Compilation can add startup time. Leave the process running while these steps complete. Progress images and saved checkpoints appear in `output/my-flux3-lora`; use the progress images to judge how well the model is learning your subject.

For `klein-base`, use 50 validation steps and guidance 3.5. To train the distilled model instead, select `model_flavour: "klein"`; this loads `black-forest-labs/flux-3-image-klein` and uses 16 validation steps with guidance 1. Accept access to that repository before starting.

## Edit and reference training

Reference conditioning is optional for both flavours. The target dataset contains the desired edited outputs and their instruction captions; conditioning datasets contain the source/reference images. Save the following as a separate dataloader JSON, point `data_backend_config` at it, and start with `train_batch_size: 1`:

```json
[
  {
    "id": "edits",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "datasets/edits",
    "caption_strategy": "textfile",
    "resolution": 512,
    "resolution_type": "pixel",
    "conditioning_data": ["references"],
    "cache_dir_vae": "cache/vae/flux3-image/edits",
    "text_embeds": "text"
  },
  {
    "id": "references",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "datasets/references",
    "conditioning_type": "reference_loose",
    "resolution": 512,
    "resolution_type": "pixel",
    "cache_dir_vae": "cache/vae/flux3-image/references"
  },
  {
    "id": "text",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux3-image-edit"
  }
]
```

Match filenames across the target and reference directories, for example `edits/sample.png`, `edits/sample.txt`, and `references/sample.png`. `reference_loose` permits different reference sizes and aspect ratios; use `reference_strict` for aligned crops. Add further conditioning dataset IDs to `conditioning_data` for multiple references, up to ten per sample; use `conditioning_multidataset_sampling: "combined"` to supply all of them together. Without `conditioning_data`, training is text-to-image. Keep each reference cache separate from target caches and caches for other model families.

With `conditioning_multidataset_sampling: "random"`, each training sample uses one reference dataset and that dataset’s caption. Set `caption_strategy: "textfile"` on each reference dataset and put its editing instruction in the matching `.txt` file, such as `references/sample.txt`.

## Generate an image with your LoRA

Find the saved checkpoint directory containing `pytorch_lora_weights.safetensors`. Substitute that directory for `output/my-flux3-lora/checkpoint-1000` below, and use the same model flavour you trained. This example uses the base model and CFG-Zero-Star guidance:

```python
import torch
from simpletuner.helpers.models.flux3_image.pipeline import Flux3ImagePipeline

pipe = Flux3ImagePipeline.from_pretrained(
    "black-forest-labs/flux-3-image-klein-base", torch_dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights("output/my-flux3-lora/checkpoint-1000")
image = pipe(
    "A photo of TOK, a small ceramic owl, sitting on a wooden shelf",
    num_inference_steps=50,
    guidance_scale=3.5,
    use_cfg_zero_star=True,
).images[0]
image.save("my-flux3-image.png")
```

## Assistant LoRA

LoRA training with `model_flavour: "klein"` automatically loads the [published training assistant](https://huggingface.co/SimpleTuner/flux-3-klein-image-training-assistant) as a frozen adapter. `assistant_lora_strength` defaults to `1.0` during training; `assistant_lora_inference_strength` defaults to `0.0` for validation. Set `disable_assistant_lora: true` to turn it off. To use another adapter, set `assistant_lora_path` and, if needed, `assistant_lora_weight_name`.

To train your own assistant, select `flux3_image-assistant-lora.peft-lora`. This recipe disables the existing assistant and learns from model-generated samples. Afterwards, set `assistant_lora_path` to its saved adapter directory in your distilled LoRA configuration.

For additional training settings, see the [options reference](../OPTIONS.md). Enable TwinFlow with both `twinflow_enabled: true` and `use_ema: true`; EMA maintains the teacher model used during this training method.
