## Qwen Image 2.1

Qwen Image 2.1 is the default (`model_flavour: "v2.1"`), using `Qwen/Qwen-Image-2.1`. It has a 32-block transformer, a Qwen3-VL text encoder, and a 64-channel VAE with 16× spatial compression.

Qwen Image 2.1 uses [Ollin’s texture-fixed VAE](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1) by default for validation and other VAE decoding. Only the decoder was fine-tuned; the encoder is unchanged, so existing 2.1 training latents remain compatible. The VAE revision is pinned independently of the base model. To reproduce outputs with the original VAE, set `pretrained_vae_model_name_or_path: "Qwen/Qwen-Image-2.1"`. Explicit VAE overrides and older flavours keep their existing behaviour.

The `qwen_image.peft-lora` example trains on `RareConcepts/Domokun` at 512px with the trigger `🟫`. Start with BF16 (`base_model_precision: "no_change"`) and use gradient checkpointing when memory is limited. The example uses separate 2.1 latent and text caches; do not reuse caches from older flavours.

```bash
simpletuner train example=qwen_image.peft-lora
```

For validation, use `validation_guidance: 1.0`, `validation_guidance_real: 1.0`, and `validation_num_inference_steps: 40`. Keep the trigger in validation prompts to check whether the subject was learned.

Qwen Image 2.1 decodes single images without retaining unused temporal feature caches. In an isolated BF16 H200 decode of one 2048×2048 image, this reduced peak allocated memory from 26.87 to 15.28 GiB with identical output. Tiled decoding saves more memory but can introduce colour seams by limiting spatial context; removing the unused caches does not fix those seams.

Earlier flavours remain available: `v1.0` selects Qwen-Image, `v2.0` selects Qwen-Image-2512, and the `edit-*` flavours keep their existing checkpoints. Their adapters and latent caches are not interchangeable with 2.1.

The 250-step Domokun recipe is a throughput example, not a reliable convergence recipe. An earlier checkpoint produced recognizable Domokun images after reloading, but fresh 250-step runs did not reproduce that result. Controls retaining padding masks, disabling compilation, and restoring the earlier RoPE expression also failed. Cached latents decode to the correct subject. The cause of the training deterioration remains unresolved; the timing tables do not establish comparable image quality across attention backends.

### VRAM presets

These examples use BF16, rank-32 LoRA, Optimi Lion and regional compilation at 512px, without gradient checkpointing. Compilation has a first-run cost; compare warm training steps. The 24 GB and 32 GB budgets were checked on L40S, not on separate 24 GB or 32 GB cards.

| VRAM budget | Example | Dataset batch size | Peak VRAM (GiB) | Warm step (s) |
| --- | --- | --- | --- | --- |
| 24 GB | `qwen_image-2.1-24g.peft-lora` | 1 | 20.6 | 0.238 |
| 32 GB | `qwen_image-2.1-32g.peft-lora` | 2 | 26.5 | 0.390 |
| 48 GB | `qwen_image-2.1-48g.peft-lora` | 2 | 26.5 | 0.390 |
| 80 GB | `qwen_image-2.1-80g.peft-lora` | 10 | 71.8 | 0.639 |
| 144 GB | `qwen_image-2.1-144g.peft-lora` | 20 | 128.4 | 1.223 |

Measured on L40S (24/32/48 GB presets), H100 (80 GB) and H200 (144 GB), using 20 steps with the first five excluded from timing. Peak VRAM includes setup. These are 512px, batch-specific measurements, not guarantees for larger images or longer prompts.

The 48 GB preset also uses batch 2: on L40S it delivered better throughput per image than batches 3, 4 and 5. Batch 5 fitted in 43.3 GiB but took 0.991 s/step, compared with 0.390 s/step at batch 2.

```bash
simpletuner train example=qwen_image-2.1-48g.peft-lora
```

Use the matching dataset file bundled with each example: its batch size is explicit. Start a fresh run when changing batch size or dataset settings; do not reuse an incompatible training-state checkpoint.

For lower memory use, enable `gradient_checkpointing: true` and `gradient_checkpointing_interval: 2`. This now checkpoints contiguous two-block groups. See the [Qwen Image 2.1 checkpoint and attention measurements](../experimental/SEGMENTED_CHECKPOINTING.md#qwen-image-21) for the measured tradeoffs; the earlier every-other-block result is superseded. BF16 fits these presets without an int8 checkpoint.

The text-to-image path avoids tensor-dependent sequence assembly so it can be captured without graph breaks. Real-valued RoPE allows Inductor to fuse normalization and rotation; the modulation, residual and MLP epilogues are also compiled. The existing Hopper CuTe ConvRot GEMM and inference-only LTX RoPE kernels are not used by these training examples.


### Experimental AnyFlow pilot

The three `qwen_image-2.1-anyflow-stage*.peft-lora` examples test whether interval distillation can retain the base model's behavior while introducing `🟫`. They are experimental; Qwen's model card does not establish that guidance distillation caused the earlier deterioration.

All stages use 1024px, BF16, AdamW, rank 32 and batch 4. Stage 1 disables gradient checkpointing on H200; stages 2 and 3 checkpoint contiguous two-block groups. This is a different workload from the 512px throughput presets above. Install the `webshart` dependency before running them.

These AnyFlow examples explicitly use `grad_clip_method: "value"` with `max_grad_norm: 0.01`: each gradient element is clamped to ±0.01. This is not a global-norm cap. To test norm clipping at 1.0, set both `grad_clip_method: "norm"` and `max_grad_norm: 1.0`.

1. **Stage 1:** 10,000 forward AnyFlow updates at `1e-5`. CC12M uses `webshart/cc12m-structured-captions` with `caption_key: "long_caption"`; e621 uses `webshart/e621-2024-webp-4Mpixel-webshart-indices`. Each prior dataset is capped at 4,096 accepted images. Their sampling weights are 0.49 each; `RareConcepts/Domokun` uses the `🟫` trigger, weight 0.02, and `repeats: 0`.
2. **Stage 2:** 2,000 on-policy DMD updates at `2e-6`, co-training the forward objective on the same mixture. This is AnyFlow DMD, not preference-pair DPO. Including the character in both stages provides real examples; the frozen teacher alone cannot teach it.
3. **Stage 3:** 100 updates at `5e-7`, with Domokun weight 0.5 and two regularisation datasets at 0.25 each, capped at 64 images each. All intervals use `r=t` and the raw flow target, retaining the interval embedder while performing a short supervised refinement. Regularisation batches use the adapter-disabled base prediction.

Review the 2,000-, 5,000- and 10,000-update checkpoints before extending stage 1 toward 20,000 updates. Stage 2 has a 2,000-update budget; stop earlier if matched validation images deteriorate. Step count alone does not establish a final model.

Dynamic compilation is enabled for variable caption lengths. `schedule_shift: 2.000802574061872` matches the released scheduler at 1024px (4,096 latent tokens); recalculate it when changing resolution.

```bash
simpletuner train example=qwen_image-2.1-anyflow-stage1.peft-lora
simpletuner train example=qwen_image-2.1-anyflow-stage2.peft-lora
simpletuner train example=qwen_image-2.1-anyflow-stage3.peft-lora
```

Stages 2 and 3 load the preceding stage's final adapter with `init_lora` and start fresh optimizer and dataloader state. Sampling weights control interleaving while datasets remain available; they are not guaranteed final sample percentages. The capped pilot does not cover either full prior corpus. The string `long_caption` field works with the existing Webshart caption selector and does not require native JSON-object caption support.

Stages 1 and 2 use `diffusion_target: "base_prediction"` and `fuse_guidance_scale: 1.0`: the diffusion branch preserves the frozen conditional field, while the other branches learn from the data. This is a preservation hypothesis, not a guarantee of character learning. Compare the supplied character and prior prompts at 4, 16 and 40 inference steps against a 40-step base-model reference; automatic validation uses 4 steps. See [AnyFlow](../experimental/ANYFLOW.md) for the objective and checkpoint requirements.

Completed pilot: stage 1 reached 10,000 updates and stage 2 reached 2,000. Freshly reloaded 40-step fox and character-prompt images remained coherent, but character prompts produced people rather than Domokun. Four-step images stayed blurred or noisy. A separate L40S comparison used both checkpoints at 1024px, seed 42 and CFG 1/2/4/6 with an empty negative prompt. Forward-call assertions verified two passes per step above CFG 1. Reviewed fox and beach samples were not rescued: higher CFG increased saturation and artifacts. These results do not identify the cause; stage 3 has not been validated.

Two stage-1 continuations from the same checkpoint each added 1,000 updates, comparing value-clipping thresholds of 0.01 and 1.0. Fresh four-step fox images remained noisy in both; 40-step beach images still depicted people. Clipping activated in 65/1,000 updates at 0.01 and 0/1,000 at 1.0. Both branches used the uncorrected optimizer, and their early gradients differed before clipping activated, so small differences cannot be attributed solely to the threshold.

<a id="qwen21-optimizer-correction"></a>

The initial stage-1 and assistant pilots predate a correction to AdamW BF16’s stochastic-add helper: it computed `other + alpha * input` instead of `input + alpha * other`. At β₁ = 0.9, the first moment therefore followed `m = 0.09 * m + g` rather than `m = 0.9 * m + 0.1 * g`. Exact-arithmetic regressions cover CPU, MPS and CUDA. The image observations remain valid for those checkpoints, but they are not a clean test of the architecture or distillation objective; training must be rechecked with the corrected optimizer.

Repeating the 1,000-update continuation with the corrected optimizer, the same starting checkpoint and value clipping at 1.0 did not rescue the reviewed four-step fox or beach images. At 40 steps the fox remained coherent, while the beach prompt still produced a person. This tests recovery of the existing checkpoint, not training from scratch with the corrected optimizer; the cause of the overall failure remains unresolved.

### Legacy Qwen Image setup (v1.0 / v2.0)

> 🆕 Looking for the edit checkpoints? See the [Qwen Image Edit quickstart](./QWEN_EDIT.md) for paired-reference training instructions.

In this example, we'll be training a LoRA for Qwen Image, a 20B parameter vision-language model. Due to its size, we'll need aggressive memory optimization techniques.

A 24GB GPU is the absolute minimum, and even then you'll need extensive quantization and careful configuration. 40GB+ is strongly recommended for a smoother experience.

When training on 24G, validations will run out of memory unless you use lower resolution or aggressive quant level beyond int8.

### Hardware requirements

Qwen Image is a 20B parameter model with a sophisticated text encoder that alone consumes ~16GB VRAM before quantization. The model uses a custom VAE with 16 latent channels.

**Important limitations:**
- **Not supported on AMD ROCm or MacOS** due to lack of efficient flash attention
- Batch size > 1 is not currently working correctly; use gradient accumulation instead
- TREAD (Text-Representation Enhanced Adversarial Diffusion) is not yet supported

### Prerequisites

Make sure that you have python installed; SimpleTuner does well with 3.10 through 3.13.

You can check this by running:

```bash
python --version
```

If you don't have python 3.13 installed on Ubuntu, you can try the following:

```bash
apt -y install python3.13 python3.13-venv
```

#### Container image dependencies

For Vast, RunPod, and TensorDock (among others), the following will work on a CUDA 12.2-12.8 image to enable compiling of CUDA extensions:

```bash
apt -y install nvidia-cuda-toolkit
```

### Installation

Install SimpleTuner via pip:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

For manual installation or development setup, see the [installation documentation](../INSTALL.md).

### Setting up the environment

To run SimpleTuner, you will need to set up a configuration file, the dataset and model directories, and a dataloader configuration file.

#### Configuration file

An experimental script, `configure.py`, may allow you to entirely skip this section through an interactive step-by-step configuration. It contains some safety features that help avoid common pitfalls.

**Note:** This doesn't configure your dataloader. You will still have to do that manually, later.

To run it:

```bash
simpletuner configure
```

> ⚠️ For users located in countries where Hugging Face Hub is not readily accessible, you should add `HF_ENDPOINT=https://hf-mirror.com` to your `~/.bashrc` or `~/.zshrc` depending on which `$SHELL` your system uses.

If you prefer to manually configure:

Copy `config/config.json.example` to `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

There, you will possibly need to modify the following variables:

- `model_type` - Set this to `lora`.
- `lora_type` - Set this to `standard` for PEFT LoRA or `lycoris` for LoKr.
- `model_family` - Set this to `qwen_image`.
- `model_flavour` - Set this to `v1.0`.
- `output_dir` - Set this to the directory where you want to store your checkpoints and validation images. It's recommended to use a full path here.
- `train_batch_size` - Set this based on your available VRAM. Current SimpleTuner Qwen overrides support batch sizes greater than 1.
- `gradient_accumulation_steps` - Set this to 2-8 if you want a larger effective batch without increasing per-step VRAM.
- `validation_resolution` - You should set this to `1024x1024` or lower for memory constraints.
  - 24G cannot handle 1024x1024 validations currently - you'll need to reduce the size
  - Other resolutions may be specified using commas to separate them: `1024x1024,768x768,512x512`
- `validation_guidance` - Use a value around 3.0-4.0 for good results.
- `validation_num_inference_steps` - Use somewhere around 30.
- `use_ema` - Setting this to `true` will help obtain smoother results but uses more memory.

- `optimizer` - Use `optimi-lion` for good results, or `adamw-bf16` if you have memory to spare.
- `mixed_precision` - Must be set to `bf16` for Qwen Image.
- `gradient_checkpointing` - **Required** to be enabled (`true`) for reasonable memory usage.
- `base_model_precision` - **Strongly recommended** to set to `int8-quanto` or `nf4-bnb` for 24GB cards.
- `quantize_via` - Set to `cpu` to avoid OOM during quantization on smaller GPUs.
- `quantize_activations` - Keep this `false` to maintain training quality.

Memory optimization settings for 24GB GPUs:
- `lora_rank` - Use 8 or lower.
- `lora_alpha` - Match this to your lora_rank value.
- `flow_schedule_shift` - Set to 1.73 (or experiment between 1.0-3.0).

Your config.json will look something like this for a minimal setup:

<details>
<summary>View example config</summary>

```json
{
    "model_type": "lora",
    "model_family": "qwen_image",
    "model_flavour": "v1.0",
    "lora_type": "standard",
    "lora_rank": 8,
    "lora_alpha": 8,
    "output_dir": "output/models-qwen_image",
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "validation_resolution": "1024x1024",
    "validation_guidance": 4.0,
    "validation_num_inference_steps": 30,
    "validation_seed": 42,
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_step_interval": 100,
    "vae_batch_size": 1,
    "seed": 42,
    "resume_from_checkpoint": "latest",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "report_to": "tensorboard",
    "optimizer": "optimi-lion",
    "num_train_epochs": 0,
    "num_eval_images": 1,
    "mixed_precision": "bf16",
    "minimum_image_size": 0,
    "max_train_steps": 1000,
    "max_grad_norm": 0.01,
    "lr_warmup_steps": 100,
    "lr_scheduler": "constant_with_warmup",
    "learning_rate": "1e-4",
    "gradient_checkpointing": "true",
    "base_model_precision": "int2-quanto",
    "quantize_via": "cpu",
    "quantize_activations": false,
    "flow_schedule_shift": 1.73,
    "disable_benchmark": false,
    "data_backend_config": "config/qwen_image/multidatabackend.json",
    "checkpoints_total_limit": 5,
    "checkpoint_step_interval": 500,
    "caption_dropout_probability": 0.0,
    "aspect_bucket_rounding": 2
}
```
</details>

> ℹ️ Multi-GPU users can reference [this document](../OPTIONS.md#environment-configuration-variables) for information on configuring the number of GPUs to use.

> ⚠️ **Critical for 24GB GPUs**: The text encoder alone uses ~16GB VRAM. With `int2-quanto` or `nf4-bnb` quantization, this can be reduced significantly.

For a quick sanity check with a known working configuration:

**Option 1 (Recommended - pip install):**
```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130

simpletuner train example=qwen_image.peft-lora
```

**Option 2 (Git clone method):**
```bash
simpletuner train env=examples/qwen_image.peft-lora
```

**Option 3 (Legacy method - still works):**
```bash
ENV=examples/qwen_image.peft-lora ./train.sh
```

### Advanced Experimental Features

<details>
<summary>Show advanced experimental details</summary>


SimpleTuner includes experimental features that can significantly improve training stability and performance.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduces exposure bias and improves output quality by letting the model generate its own inputs during training.

> ⚠️ These features increase the computational overhead of training.

#### Validation prompts

Inside `config/config.json` is the "primary validation prompt", which is typically the main instance_prompt you are training on for your single subject or style. Additionally, a JSON file may be created that contains extra prompts to run through during validations.

The example config file `config/user_prompt_library.json.example` contains the following format:

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

The nicknames are the filename for the validation, so keep them short and compatible with your filesystem.

To point the trainer to this prompt library, add it to your config.json:
```json
  "validation_prompt_library": "config/user_prompt_library.json",
```

A set of diverse prompts will help determine whether the model is learning properly:

```json
{
    "anime_style": "a breathtaking anime-style portrait with vibrant colors and expressive features",
    "chef_cooking": "a high-quality, detailed photograph of a sous-chef immersed in culinary creation",
    "portrait": "a lifelike and intimate portrait showcasing unique personality and charm",
    "cinematic": "a cinematic, visually stunning photo with dramatic and captivating presence",
    "elegant": "an elegant and timeless portrait exuding grace and sophistication",
    "adventurous": "a dynamic and adventurous photo captured in an exciting moment",
    "mysterious": "a mysterious and enigmatic portrait shrouded in shadows and intrigue",
    "vintage": "a vintage-style portrait evoking the charm and nostalgia of a bygone era",
    "artistic": "an artistic and abstract representation blending creativity with visual storytelling",
    "futuristic": "a futuristic and cutting-edge portrayal set against advanced technology"
}
```

#### CLIP score tracking

If you wish to enable evaluations to score the model's performance, see [this document](../evaluation/CLIP_SCORES.md) for information on configuring and interpreting CLIP scores.

#### Stable evaluation loss

If you wish to use stable MSE loss to score the model's performance, see [this document](../evaluation/EVAL_LOSS.md) for information on configuring and interpreting evaluation loss.

#### Validation previews

SimpleTuner supports streaming intermediate validation previews during generation using Tiny AutoEncoder models. This allows you to see validation images being generated step-by-step in real-time via webhook callbacks.

To enable:
```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```

**Requirements:**
- Webhook configuration
- Validation enabled

Set `validation_preview_steps` to a higher value (e.g., 3 or 5) to reduce Tiny AutoEncoder overhead. With `validation_num_inference_steps=20` and `validation_preview_steps=5`, you'll receive preview images at steps 5, 10, 15, and 20.

#### Flow schedule shifting

Qwen Image, as a flow-matching model, supports timestep schedule shifting to control which parts of the generation process are trained.

The `flow_schedule_shift` parameter controls this:
- Lower values (0.1-1.0): Focus on fine details
- Medium values (1.0-3.0): Balanced training (recommended)
- Higher values (3.0-6.0): Focus on large compositional features

##### Auto-shift
You can enable resolution-dependent timestep shift with `--flow_schedule_auto_shift`, which uses higher shift values for larger images and lower shift values for smaller images. This can provide stable but potentially mediocre training results.

##### Manual specification
A `--flow_schedule_shift` value of 1.73 is recommended as a starting point for Qwen Image, though you may need to experiment based on your dataset and goals.

#### Dataset considerations

It's crucial to have a substantial dataset to train your model on. There are limitations on the dataset size, and you will need to ensure that your dataset is large enough to train your model effectively.

> ℹ️ With few enough images, you might see a message **no images detected in dataset** - increasing the `repeats` value will overcome this limitation.

> ⚠️ **Important**: Due to current limitations, keep `train_batch_size` at 1 and use `gradient_accumulation_steps` instead to simulate larger batch sizes.

Create a `--data_backend_config` (`config/multidatabackend.json`) document containing this:

```json
[
  {
    "id": "pseudo-camera-10k-qwen",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 512,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/qwen_image/pseudo-camera-10k",
    "instance_data_dir": "datasets/pseudo-camera-10k",
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "discovery",
    "repeats": 0,
    "is_regularisation_data": true
  },
  {
    "id": "dreambooth-subject",
    "type": "local",
    "crop": false,
    "resolution": 1024,
    "minimum_image_size": 512,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/qwen_image/dreambooth-subject",
    "instance_data_dir": "datasets/dreambooth-subject",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/qwen_image",
    "disabled": false,
    "write_batch_size": 16
  }
]
```

> ℹ️ Use `caption_strategy=textfile` if you have `.txt` files containing captions.
> See caption_strategy options and requirements in [DATALOADER.md](../DATALOADER.md#caption_strategy).
> ℹ️ Note the reduced `write_batch_size` for text embeds to avoid OOM issues.

Then, create a `datasets` directory:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

This will download about 10k photograph samples to your `datasets/pseudo-camera-10k` directory, which will be automatically created for you.

Your Dreambooth images should go into the `datasets/dreambooth-subject` directory.

#### Login to WandB and Huggingface Hub

You'll want to login to WandB and HF Hub before beginning training, especially if you're using `--push_to_hub` and `--report_to=wandb`.

If you're going to be pushing items to a Git LFS repository manually, you should also run `git config --global credential.helper store`

Run the following commands:

```bash
wandb login
```

and

```bash
huggingface-cli login
```

Follow the instructions to log in to both services.

</details>

### Executing the training run

From the SimpleTuner directory, one simply has to run:

```bash
./train.sh
```

This will begin the text embed and VAE output caching to disk.

For more information, see the [dataloader](../DATALOADER.md) and [tutorial](../TUTORIAL.md) documents.

### Memory optimization tips

#### Lowest VRAM config (24GB minimum)

The lowest VRAM Qwen Image configuration requires approximately 24GB:

- OS: Ubuntu Linux 24
- GPU: A single NVIDIA CUDA device (24GB minimum)
- System memory: 64GB+ recommended
- Base model precision:
  - For NVIDIA systems: `int2-quanto` or `nf4-bnb` (required for 24GB cards)
  - `int4-quanto` can work but may have lower quality
- Optimizer: `optimi-lion` or `bnb-lion8bit-paged` for memory efficiency
- Resolution: Start with 512px or 768px, work up to 1024px if memory allows
- Batch size: 1 (mandatory due to current limitations)
- Gradient accumulation steps: 2-8 to simulate larger batches
- Enable `--gradient_checkpointing` (required)
- Use `--quantize_via=cpu` to avoid OOM during startup
- Use a small LoRA rank (1-8)
- Setting the environment variable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps minimize VRAM usage

**NOTE**: Pre-caching of VAE embeds and text encoder outputs will use significant memory. Enable `offload_during_startup=true` if you encounter OOM issues.

### Running inference on the LoRA afterward

Since Qwen Image is a newer model, here's a functioning example for inference:

<details>
<summary>Show Python inference example</summary>

```python
import torch
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration

model_id = 'Qwen/Qwen-Image'
adapter_id = 'your-username/your-lora-name'

# Load the pipeline
pipeline = QwenImagePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)

# Load LoRA weights
pipeline.load_lora_weights(adapter_id)

# Optional: quantize the model to save VRAM
from optimum.quanto import quantize, freeze, qint8
quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)

# Move to device
pipeline.to('cuda' if torch.cuda.is_available() else 'cpu')

# Generate an image
prompt = "Your test prompt here"
negative_prompt = 'ugly, cropped, blurry, low-quality, mediocre average'

image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=4.0,
    generator=torch.Generator(device='cuda').manual_seed(42),
    width=1024,
    height=1024,
).images[0]

image.save("output.png", format="PNG")
```
</details>

### Notes & troubleshooting tips

#### Batch size limitations

Older diffusers Qwen builds had batch-size > 1 issues caused by text embed padding and attention-mask handling. Current SimpleTuner Qwen overrides patch both paths, so larger batches work if your VRAM allows them.
- Increase `train_batch_size` only after confirming your memory headroom.
- If you still see artifacts on an older install, update and regenerate any stale text embeds.

#### Quantization

- `int2-quanto` provides the most aggressive memory savings but may impact quality
- `nf4-bnb` offers a good balance between memory and quality
- `int4-quanto` is a middle ground option
- Avoid `int8` unless you have 40GB+ VRAM

#### Learning rates

For LoRA training:
- Small LoRAs (rank 1-8): Use learning rates around 1e-4
- Larger LoRAs (rank 16-32): Use learning rates around 5e-5
- With Prodigy optimizer: Start with 1.0 and let it adapt

#### Image artifacts

If you encounter artifacts:
- Lower your learning rate
- Increase gradient accumulation steps
- Ensure your images are high quality and properly preprocessed
- Consider using lower resolutions initially

#### Multiple-resolution training

Start training at lower resolutions (512px or 768px) to speed up initial learning, then fine-tune at 1024px. Enable `--flow_schedule_auto_shift` when training at different resolutions.

### Platform limitations

**Not supported on:**
- AMD ROCm (lacks efficient flash attention implementation)
- Apple Silicon/MacOS (memory and attention limitations)
- Consumer GPUs with less than 24GB VRAM

### Current known issues

1. Batch size > 1 doesn't work correctly (use gradient accumulation)
2. TREAD is not yet supported
3. High memory usage from text encoder (~16GB before quantization)
4. Sequence length handling issues ([upstream issue](https://github.com/huggingface/diffusers/issues/12075))

For additional help and troubleshooting, consult the [SimpleTuner documentation](/documentation) or join the community Discord.

<a id="assistant-lora"></a>

### Training an assistant LoRA from captions

`qwen_image-2.1-assistant-lora.peft-lora` is an experimental L40S starting point: BF16, batch 1, interval-2 checkpointing, rank 32 and AdamW BF16 at `1e-4`. It budgets 1,000 updates with validation/checkpoints every 50. Replace the twelve smoke captions with diverse captions before a substantive run; the example is not a validated convergence recipe.

`grad_clip_method: "norm"`, `max_grad_norm: 1.0`.

Set `distillation_method: assistant_lora`. The caption backend precomputes text embeddings, then each batch generates fresh base-model latents with the adapter disabled, 40 native inference steps and CFG 1. The private generation pipeline reuses the transformer without loading a VAE, processor or text encoder. The adapter is restored before ordinary denoising training. Terminal latents are never cached. This currently supports Qwen Image 2.1 text-to-image only.

`distillation_config.assistant_lora` accepts `num_inference_steps` (default 40), `resolutions` (a nonempty list of `[width, height]`, default `[[1024, 1024]]`) and `seed` (default 42). Qwen dimensions must be multiples of 32. Resolutions cycle per batch; noise seeds advance per sample, including short batches. Checkpoints preserve these counters. Resume with unchanged dataset, batch size, accumulation and distributed topology. Text-cache on-demand mode is not supported.

For a plumbing test, use 8 updates and 2 teacher steps. Restore 40 teacher steps before judging samples. Review a larger run at 100, 250, 500 and 1,000 updates; compare adapter-enabled images against the same base-model prompts/seeds. A useful assistant must still be tested in a separate concept-training run.

[Ostris describes training on the model's own generated images at a low learning rate](https://huggingface.co/ostris/zimage_turbo_training_adapter). This trains the positive adapter. In a subsequent concept run, set `assistant_lora_path` to the saved adapter and enable assistant loading; SimpleTuner keeps it frozen during training and removes it during sampling. No default Qwen 2.1 assistant is downloaded. Its benefit for Qwen 2.1 remains an experiment.

Use this as the sole distillation method; composition with other distillers is not supported.

Caption datasets require `dataloader_prefetch: false` so checkpoint cursors represent consumed captions. Resume rejects changes to caption identities/text, batch size, repeats, shuffle, seed, accumulation or distributed layout. Assistant checkpoints also reject changes to the generation seed, resolution list or teacher inference-step count.

<a id="assistant-lora-multires"></a>

#### Assistant experiment with four base resolutions and aspect ratio buckets

`qwen_image-2.1-assistant-lora-multires.peft-lora` cycles through 12 aspect ratio buckets across the 512, 1024, 1536 and 2048 base resolutions. Each base has square, 4:7 portrait and 7:4 landscape buckets; each batch uses one bucket, with equal exposure per base and aspect over a complete cycle. It keeps the base example’s 40 teacher steps, BF16 batch 1, interval-2 checkpointing, rank 32, AdamW BF16 and 1,000-update budget, and shares its caption backend and validation prompts. Replace the smoke captions with the same diverse caption set used for the baseline. Start with a fresh adapter and optimizer in the new output directory; `resume_from_checkpoint: ""` disables resume. Preserve the first run’s weights for comparison.

This tests whether exposure to multiple image sizes improves the assistant; it does not establish a native-resolution requirement or a quality benefit. The 1,000-update L40S run completed across all 12 buckets, with validation at 1024×1024 and 2048×2048. Final fox and portrait images remained coherent, with the known colour seams from tiled VAE decoding. Teacher generation uses latent targets without VAE decoding. Downstream concept-training benefit remains unverified.

Assistant LoRA uses the same scoped minimum of 32 Dynamo cache entries as AnyFlow, accommodating teacher, student and validation variants. Larger user limits are preserved, and the original limit is restored when the run exits. Monitor recompilations when adding resolutions or longer captions.

A subsequent matched Domokun screen trained two fresh adapters for 250 updates each at 2048px, batch 1, LR `1e-5` and value clipping at 1.0. Assistant training strength was 0 for the control and 1 for the assisted run; inference disabled the assistant in both. Initial adapter weights and validation images matched exactly. At 40 inference steps and 1024px, both final runs still produced people for the character prompts and coherent fox/portrait priors; no assistant benefit was demonstrated. Both runs and the assistant preparation used the uncorrected optimizer described [above](#qwen21-optimizer-correction).

<a id="assistant-lora-offline"></a>

#### Assistant LoRA from reusable generated images

`qwen_image-2.1-assistant-lora-offline.peft-lora` trains on [10,000 generated images](https://huggingface.co/datasets/webshart/qwen-image-2.1-generated-images) through Webshart. The dataset uses CC12M `long_caption` prompts, 40 native teacher steps, CFG 1 and full-frame VAE decoding. Twelve image backends cover square, portrait and landscape buckets at 512, 1024, 1536 and 2048 base resolutions, with equal sampling weights and no repeats.

This fresh-run recipe uses the corrected `adamw_bf16`, LR `1e-4`, `grad_clip_method: "norm"`, `max_grad_norm: 1.0`, BF16 batch 1, rank 32 and interval-2 checkpointing. It budgets 1,000 updates, saves every 50 and validates every 100 at 1024. Render separate 2048px previews before publishing. VAE tiling is disabled and VAE encoding uses batch 1. `vae_cache_ondemand: true` encodes and caches images as they are sampled, avoiding a full 10,000-image encoding pass before 1,000 updates. Ordinary image training replaces online teacher generation; omit `distillation_method` and keep `disable_assistant_lora: true` while creating the new assistant.

The dataset can be reused across experiments. PNG targets require VAE re-encoding, so this is not identical to training on the teacher's terminal latents. Inspect validation images before publishing an adapter and test its benefit in a separate concept run. Start fresh when switching from the caption-only recipe; do not resume its optimizer or dataset state.

The [replacement v1 assistant](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v1) completed this 1,000-update recipe on L40S, with final images reviewed at 1024 and 2048. A matched 1,000-update Domokun comparison at 2048, LR `1e-4` and norm clipping 1.0 kept the assistant frozen during training and disabled it for validation. Both control and assisted runs still produced people for the two character prompts. The control leaked strong Domokun features into an unrelated fox prompt; the assisted run retained a recognizable fox. Both retained a coherent portrait. This is limited evidence of reduced spillover, not successful concept learning or a general quality benefit; the evaluation uses one training seed and four prompts.

```bash
simpletuner train example=qwen_image-2.1-assistant-lora-offline.peft-lora
```
