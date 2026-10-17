# MOSS-Image quickstart

Train a MOSS-Image LoRA on your own images. A LoRA is a small adapter that teaches the image model a subject or style while keeping the original checkpoint separate.

Choose `model_family: "moss_image"`, `model_flavour: "base"`, and `model_type: "lora"` to start with [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base). Choose `model_flavour: "sft"` for [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT), which was refined on curated data. Both share a 5B image transformer, a Qwen3.5-9B text encoder, and a FLUX.2 VAE. Both checkpoints are released under [Apache-2.0](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/blob/main/LICENSE). The text encoder stays frozen during training.

## Choose a VRAM preset

Choose the recipe matching your GPU memory. Every tier trains at a 1024×1024 equivalent area, preserves image aspect ratios, and keeps rank-32 BF16 LoRA and REPA enabled.

| GPU memory | Base example | SFT example | Images per batch | `gradient_checkpointing_interval` |
| --- | --- | --- | ---: | ---: |
| 24 GB | `moss_image-24g.peft-lora` | `moss_image-sft-24g.peft-lora` | 1 | 1 |
| 32 GB | `moss_image-32g.peft-lora` | `moss_image-sft-32g.peft-lora` | 1 | 2 |
| 48 GB | `moss_image-48g.peft-lora` | `moss_image-sft-48g.peft-lora` | 2 | 2 |
| 80 GB | `moss_image-80g.peft-lora` | `moss_image-sft-80g.peft-lora` | 4 | 2 |
| 144 GB | `moss_image-144g.peft-lora` | `moss_image-sft-144g.peft-lora` | 8 | 2 |

The default `moss_image.peft-lora` and `moss_image-sft.peft-lora` examples use the 48 GB settings. Batch sizes are starting points: use a smaller `train_batch_size` for longer captions, additional training features, or a dataset with fewer images in an aspect bucket than the batch size. The 144 GB tier is intended for H200-class GPUs.

## Install SimpleTuner

Use Python 3.13 and an NVIDIA GPU for the CUDA recipe below. See the [installation guide](../INSTALL.md) for other platforms and CUDA versions.

```bash
git clone https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[cuda,attention-cute]'
```

## Prepare images and captions

Create `datasets/moss-image` and add images with matching `.txt` captions. Use sharp images with enough detail for training at roughly one megapixel. A square image at this scale is 1024×1024; wide and tall images keep their aspect ratios.

```text
datasets/moss-image/
  image01.png
  image01.txt
  image02.jpg
  image02.txt
```

Choose a subject word such as `TOK` and include it in every caption for that subject. Describe the visible subject, composition, lighting, setting, and style. English is the project’s recommended starting language. Copy visible text accurately, including intended line breaks. For example: `A photograph of TOK, a small blue ceramic owl on a wooden table beside a window.`

## Set up your training configuration

Copy the 48 GB Base recipe and its sample prompts. Replace `48g` with your selected tier. For SFT, replace `moss_image-48g.peft-lora` with `moss_image-sft-48g.peft-lora` in the commands.

```bash
mkdir -p config/examples datasets/moss-image
cp -R simpletuner/examples/moss_image-48g.peft-lora config/examples/
cp simpletuner/examples/moss_image-prompts.json config/examples/
cp config/examples/moss_image-48g.peft-lora/config.json config/config.json
```

The supplied configuration uses rank-32 BF16 LoRA, learning rate `1e-4`, gradient checkpointing, REPA, 4,000 optimizer steps, and 1024-scale training. Open `config/config.json` and edit these entries inside the existing configuration:

```json
{
  "output_dir": "output/my-moss-lora",
  "validation_prompt": "A studio photograph of TOK on a wooden table, soft window light, a pale gray background.",
  "user_prompt_library": null
}
```

The copied `dataloader.json` points to `datasets/moss-image`, reads matching text files, and caches image latents and text embeddings. Its `resolution: 1.048576` with `resolution_type: "area"` means a 1024×1024 equivalent area. Image dimensions are aligned to multiples of 16. See the [dataloader guide](../DATALOADER.md) for more datasets and regularisation images.

## Train and compare previews

Run `simpletuner train` from the SimpleTuner directory with the virtual environment active. The first run downloads the model and REPA encoder and builds caches. The recipe saves checkpoints and validation images every 250 steps. Use the previews to compare subject learning, composition, and detail; adjust `max_train_steps` to suit your dataset.

```bash
simpletuner train
```

## Sampling settings

Start with **50 steps**, **CFG 3**, and **1024×1024** validation. `moss_image_validation_shift` defaults to **5** for sampling; training uses the checkpoint’s unshifted grid. `moss_image_uncond_skip_layers` defaults to **[22]**, skipping layer 22 in the negative guidance branch. Set it to `[]` for standard CFG. Additional positive-branch skip-layer guidance uses `validation_guidance_skip_layers`, `validation_guidance_skip_scale`, and the start/stop settings in [OPTIONS](../OPTIONS.md). In Python generation, `use_cfg_zero_star=True` enables CFG-Zero*.

## Preview images while sampling

To see progress frames during denoising in the WebUI, add the following settings. Tiny AutoEncoder previews use TAEF2; final validation images use the full VAE. CLI runs can deliver previews through a configured webhook.

```json
{
  "validation_preview": true,
  "validation_preview_steps": 5
}
```

## Memory and training options

The recipe sets `train_batch_size` for your chosen VRAM tier. Reduce it for a small dataset or when captions and additional features need more memory. `gradient_checkpointing_backend` accepts `torch`, `unsloth`, `torch-ffn`, and `unsloth-ffn`; `gradient_checkpointing_interval` and `gradient_checkpointing_segment_stride` control its coverage. Musubi block swapping moves selected transformer blocks between GPU and CPU using `musubi_blocks_to_swap`. RamTorch uses `ramtorch: true` and supports selective offloading with `ramtorch_transformer_percent`. These offload options use system RAM and add transfer work; configure one approach at a time. See [OPTIONS](../OPTIONS.md).

The CUDA installation includes CuTE attention. During training, the default `attention_mechanism: "kohaku-fa-auto"` selects CuTE on Ada and Hopper GPUs and native SDPA on other architectures. To offload attention tensors saved for backward, enable `gradient_checkpointing_offload_attention: true`; `gradient_checkpointing_offload_prefetch: true` enables transfer prefetching. See [OPTIONS](../OPTIONS.md) for other attention backends.

The recipe enables [REPA](../experimental/IMAGE_REPA.md). To align later image features with an earlier layer, add [LayerSync](../experimental/LAYERSYNC.md):

```json
{
  "layersync_enabled": true,
  "layersync_student_block": 20,
  "layersync_teacher_block": 8,
  "layersync_lambda": 0.2
}
```

For post-training workflows, follow the [AnyFlow](../quickstart/ANYFLOW.md) or [TwinFlow](../distillation/TWINFLOW.md) guide. Keep `lora_dropout: 0.0` for AnyFlow. When loading an AnyFlow LoRA for Python generation, pass its training settings to `load_lora_weights`, for example `flowmap_deltatime_type="r", flowmap_gate_value=0.25`.

## Prompt enhancement and layouts

MOSS-Image accepts detailed captions, tag sequences, and JSON layout prompts. The project’s [prompt-enhancer skill](https://github.com/OpenMOSS/MOSS-Image/tree/main/skills/moss-image-prompt-enhancer) helps an agent compose captions and layouts. Use its completed text as a caption or generation prompt. The [project guide](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md) includes captioning recommendations.

A layout describes the scene, rendered text, and object positions. Coordinates use `[y_min, x_min, y_max, x_max]` in pixels with the canvas centre as the origin. On a 1024×1024 canvas the range is −512 to +512. Serialize a layout with `json.dumps(layout, ensure_ascii=False)` and pass the resulting string as the prompt; use generation dimensions matching the layout.

To expand captions into native JSON during preprocessing, add these settings to your training configuration:

```json
{
  "moss_image_auto_json": true,
  "moss_image_prompt_upsample": true
}
```

The default caption model is `Qwen/Qwen3.5-27B`, one of the project’s recommended VLMs. Use an 80 GB GPU and allow extra system RAM for it. SimpleTuner uses each training image and its bucket dimensions to generate scene descriptions and centered pixel bounding boxes, then caches the text embeddings. Validation prompts are expanded too; empty and negative prompts stay unchanged. To wrap existing captions as JSON without loading a caption model, enable only `moss_image_auto_json`.

```json
{
  "summary": {"caption": "A cream travel poster with the title EXPLORE above a blue mountain."},
  "items": [
    {"label": "text", "text": "EXPLORE", "caption": "Large navy title", "bbox": [-416, -384, -288, 384]},
    {"label": "object", "caption": "Blue mountain", "bbox": [-192, -288, 384, 288]}
  ],
  "bbox_coord_space": "abs_centralized",
  "bbox_format": "yxyx"
}
```

## Generate images with your LoRA

The final adapter is `pytorch_lora_weights.safetensors` in your output directory. Load it with the same Base or SFT checkpoint used for training. Keep your subject word in the prompt and describe the scene you want:

```python
import torch
from simpletuner.helpers.models.moss_image.pipeline import MOSSImagePipeline

pipe = MOSSImagePipeline.from_pretrained(
    "OpenMOSS-Team/MOSS-Image-Base", torch_dtype=torch.bfloat16
)
pipe.load_lora_weights("output/my-moss-lora")
pipe.to("cuda")
image = pipe(
    "A studio photograph of TOK on a wooden table, soft window light, a pale gray background.",
    height=1024, width=1024, num_inference_steps=50, guidance_scale=3.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("my-moss-image.png")
```
