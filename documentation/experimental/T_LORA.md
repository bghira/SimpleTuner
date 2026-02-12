# T-LoRA (Timestep-dependent LoRA)

## Background

Standard LoRA fine-tuning applies a fixed low-rank adaptation uniformly across all diffusion timesteps. When training data is limited (especially single-image customisation), this leads to overfitting — the model memorises noise patterns at high-noise timesteps where little semantic information exists.

**T-LoRA** ([Soboleva et al., 2025](https://arxiv.org/abs/2507.05964)) solves this by dynamically adjusting the number of active LoRA ranks based on the current diffusion timestep:

- **High noise** (early denoising, $t \to T$): fewer ranks are active, preventing the model from memorising uninformative noise patterns.
- **Low noise** (late denoising, $t \to 0$): more ranks are active, allowing the model to capture fine concept details.

SimpleTuner's T-LoRA support is built on the [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) library and requires a LyCORIS version that includes the `lycoris.modules.tlora` module.

> 🟡 **Experimental:** T-LoRA with video models may produce subpar results because temporal compression blends frames across timestep boundaries.

## Quick setup

### 1. Set your training config

In your `config.json`, use LyCORIS with a separate T-LoRA config file:

```json
{
    "model_type": "lora",
    "lora_type": "lycoris",
    "lycoris_config": "config/lycoris_tlora.json",
    "validation_lycoris_strength": 1.0
}
```

### 2. Create the LyCORIS T-LoRA config

Create `config/lycoris_tlora.json`:

```json
{
    "algo": "tlora",
    "multiplier": 1.0,
    "linear_dim": 64,
    "linear_alpha": 32,
    "apply_preset": {
        "target_module": ["Attention", "FeedForward"]
    }
}
```

That is all you need to start training. The sections below cover optional tuning and inference.

## Configuration reference

### Required fields

| Field | Type | Description |
|-------|------|-------------|
| `algo` | string | Must be `"tlora"` |
| `multiplier` | float | LoRA strength multiplier. Keep at `1.0` unless you know what you are doing |
| `linear_dim` | int | LoRA rank. This becomes `max_rank` in the masking schedule |
| `linear_alpha` | int | LoRA scaling factor (separate from `tlora_alpha`) |

### Optional fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tlora_min_rank` | int | `1` | Minimum active ranks at the highest noise level |
| `tlora_alpha` | float | `1.0` | Masking schedule exponent. `1.0` is linear; values above `1.0` shift more capacity toward fine-detail steps |
| `apply_preset` | object | — | Module targeting via `target_module` and `module_algo_map` |

### Model-specific module targets

For most models the generic `["Attention", "FeedForward"]` targets work. For Flux 2 (Klein), use the custom class names:

```json
{
    "algo": "tlora",
    "multiplier": 1.0,
    "linear_dim": 64,
    "linear_alpha": 32,
    "apply_preset": {
        "target_module": [
            "Flux2Attention", "Flux2FeedForward", "Flux2ParallelSelfAttention"
        ]
    }
}
```

See the [LyCORIS documentation](../LYCORIS.md) for the full list of per-model module targets.

## Tuning knobs

### `linear_dim` (rank)

Higher rank = more parameters and expressivity, but more prone to overfitting with limited data. The original T-LoRA paper uses rank 64 for SDXL single-image customisation.

### `tlora_min_rank`

Controls the floor for rank activation at the noisiest timestep. Increasing this lets the model learn coarser structure but reduces the overfitting benefit. Start with the default of `1` and raise only if convergence is too slow.

### `tlora_alpha` (schedule exponent)

Controls the curve shape of the masking schedule:

- `1.0` — linear interpolation between `min_rank` and `max_rank`
- `> 1.0` — more aggressive masking at high noise; most ranks only activate near the end of denoising
- `< 1.0` — gentler masking; ranks activate earlier

<details>
<summary>Schedule visualisation (rank vs. timestep)</summary>

With `linear_dim=64`, `tlora_min_rank=1`, for a 1000-step scheduler:

```
alpha=1.0 (linear):
  t=0   (clean)  → 64 active ranks
  t=250 (25%)    → 48 active ranks
  t=500 (50%)    → 32 active ranks
  t=750 (75%)    → 16 active ranks
  t=999 (noise)  →  1 active rank

alpha=2.0 (quadratic — biased toward detail):
  t=0   (clean)  → 64 active ranks
  t=250 (25%)    → 60 active ranks
  t=500 (50%)    → 48 active ranks
  t=750 (75%)    → 20 active ranks
  t=999 (noise)  →  1 active rank

alpha=0.5 (sqrt — biased toward structure):
  t=0   (clean)  → 64 active ranks
  t=250 (25%)    → 55 active ranks
  t=500 (50%)    → 46 active ranks
  t=750 (75%)    → 33 active ranks
  t=999 (noise)  →  1 active rank
```

</details>

## Inference with SimpleTuner pipelines

SimpleTuner's vendored pipelines have built-in T-LoRA support. During validation, the masking parameters from training are automatically reused at each denoising step — no extra configuration is needed.

For standalone inference outside of training, you can import SimpleTuner's pipeline directly and set the `_tlora_config` attribute. This ensures the per-step masking matches what the model was trained with.

### SDXL example

```py
import torch
from lycoris import create_lycoris_from_weights

# Use SimpleTuner's vendored SDXL pipeline (has T-LoRA support built in)
from simpletuner.helpers.models.sdxl.pipeline import StableDiffusionXLPipeline
from diffusers import AutoencoderKL, EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
dtype = torch.bfloat16
device = "cuda"

# Load pipeline components
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype)
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
from diffusers import UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)

# Load and apply LyCORIS T-LoRA weights
lora_path = "path/to/pytorch_lora_weights.safetensors"
wrapper, _ = create_lycoris_from_weights(1.0, lora_path, unet)
wrapper.merge_to()

unet.to(device)

pipe = StableDiffusionXLPipeline(
    scheduler=scheduler,
    vae=vae,
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    unet=unet,
)

# Enable T-LoRA inference masking — must match training config
pipe._tlora_config = {
    "max_rank": 64,      # linear_dim from your lycoris config
    "min_rank": 1,       # tlora_min_rank (default 1)
    "alpha": 1.0,        # tlora_alpha (default 1.0)
}

with torch.inference_mode():
    image = pipe(
        prompt="a sks dog riding a surfboard",
        width=1024,
        height=1024,
        num_inference_steps=25,
        guidance_scale=5.0,
    ).images[0]

image.save("tlora_output.png")
```

### Flux example

```py
import torch
from lycoris import create_lycoris_from_weights

# Use SimpleTuner's vendored Flux pipeline (has T-LoRA support built in)
from simpletuner.helpers.models.flux.pipeline import FluxPipeline
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

bfl_repo = "black-forest-labs/FLUX.1-dev"
dtype = torch.bfloat16
device = "cuda"

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)
tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2")
vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype)
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype)

# Load and apply LyCORIS T-LoRA weights
lora_path = "path/to/pytorch_lora_weights.safetensors"
wrapper, _ = create_lycoris_from_weights(1.0, lora_path, transformer)
wrapper.merge_to()

transformer.to(device)

pipe = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=transformer,
)

# Enable T-LoRA inference masking
pipe._tlora_config = {
    "max_rank": 64,
    "min_rank": 1,
    "alpha": 1.0,
}

with torch.inference_mode():
    image = pipe(
        prompt="a sks dog riding a surfboard",
        width=1024,
        height=1024,
        num_inference_steps=25,
        guidance_scale=3.5,
    ).images[0]

image.save("tlora_flux_output.png")
```

> **Note:** You must use SimpleTuner's vendored pipeline (e.g. `simpletuner.helpers.models.flux.pipeline.FluxPipeline`), not the stock Diffusers pipeline. Only the vendored pipelines contain the per-step T-LoRA masking logic.

### Why not just use `merge_to()` and skip the masking?

`merge_to()` bakes the LoRA weights into the base model permanently — this is needed so the LoRA parameters are active during the forward pass. However, T-LoRA was **trained** with timestep-dependent rank masking: certain ranks were zeroed out depending on the noise level. Without reapplying that same masking during inference, all ranks fire at every timestep, producing over-saturated or burnt-looking images.

Setting `_tlora_config` on the pipeline tells the denoising loop to apply the correct mask before each model forward pass and clear it afterward.

<details>
<summary>How the masking works internally</summary>

At each denoising step, the pipeline calls:

```python
from simpletuner.helpers.training.lycoris import apply_tlora_inference_mask, clear_tlora_mask

_tlora_cfg = getattr(self, "_tlora_config", None)
if _tlora_cfg:
    apply_tlora_inference_mask(
        timestep=int(t),
        max_timestep=self.scheduler.config.num_train_timesteps,
        max_rank=_tlora_cfg["max_rank"],
        min_rank=_tlora_cfg["min_rank"],
        alpha=_tlora_cfg["alpha"],
    )
try:
    noise_pred = self.unet(...)  # or self.transformer(...)
finally:
    if _tlora_cfg:
        clear_tlora_mask()
```

`apply_tlora_inference_mask` computes a binary mask of shape `(1, max_rank)` using the formula:

$$r = \left\lfloor\left(\frac{T - t}{T}\right)^\alpha \cdot (R_{\max} - R_{\min})\right\rfloor + R_{\min}$$

where $T$ is the max scheduler timestep, $R_{\max}$ is `linear_dim`, and $R_{\min}$ is `tlora_min_rank`. The first $r$ elements of the mask are set to `1.0` and the rest to `0.0`. This mask is then set globally on all T-LoRA modules via `set_timestep_mask()` from LyCORIS.

After the forward pass completes, `clear_tlora_mask()` removes the mask state so it does not leak into subsequent operations.

</details>

<details>
<summary>How SimpleTuner passes config during validation</summary>

During training, the T-LoRA config dict (`max_rank`, `min_rank`, `alpha`) is stored on the Accelerator object. When validation runs, `validation.py` copies this config onto the pipeline:

```python
# setup_pipeline()
if getattr(self.accelerator, "_tlora_active", False):
    self.model.pipeline._tlora_config = self.accelerator._tlora_config

# clean_pipeline()
if hasattr(self.model.pipeline, "_tlora_config"):
    del self.model.pipeline._tlora_config
```

This is fully automatic — no user configuration is required for validation images to use the correct masking.

</details>

## Upstream: the T-LoRA paper

<details>
<summary>Paper details and algorithm</summary>

**T-LoRA: Single Image Diffusion Model Customization Without Overfitting**
Vera Soboleva, Aibek Alanov, Andrey Kuznetsov, Konstantin Sobolev
[arXiv:2507.05964](https://arxiv.org/abs/2507.05964) — Accepted to AAAI 2026

The paper introduces two complementary innovations:

### 1. Timestep-dependent rank masking

The core insight is that higher diffusion timesteps (noisier inputs) are more prone to overfitting than lower timesteps. At high noise, the latent contains mostly random noise with little semantic signal — training a full-rank adapter on this teaches the model to memorise noise patterns rather than learn the target concept.

T-LoRA addresses this with a dynamic masking schedule that restricts the active LoRA rank based on the current timestep.

### 2. Orthogonal weight parametrisation (optional)

The paper also proposes initialising LoRA weights via SVD decomposition of the original model weights, enforcing orthogonality through a regularisation loss. This ensures independence between adapter components.

SimpleTuner's LyCORIS integration focuses on the timestep masking component, which is the primary driver of the overfitting reduction. The orthogonal initialisation is part of the standalone T-LoRA implementation but is not currently used by the LyCORIS `tlora` algorithm.

### Citation

```bibtex
@misc{soboleva2025tlorasingleimagediffusion,
      title={T-LoRA: Single Image Diffusion Model Customization Without Overfitting},
      author={Vera Soboleva and Aibek Alanov and Andrey Kuznetsov and Konstantin Sobolev},
      year={2025},
      eprint={2507.05964},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.05964},
}
```

</details>

## Common pitfalls

- **Forgot `_tlora_config` during inference:** Images look over-saturated or burnt. All ranks fire at every timestep instead of following the trained masking schedule.
- **Using stock Diffusers pipeline:** The stock pipelines do not contain T-LoRA masking logic. You must use SimpleTuner's vendored pipelines.
- **`linear_dim` mismatch:** The `max_rank` in `_tlora_config` must match the `linear_dim` used during training, or the mask dimensions will be wrong.
- **Video models:** Temporal compression blends frames across timestep boundaries, which can weaken the timestep-dependent masking signal. Results may be subpar.
- **SDXL + FeedForward modules:** Training FeedForward modules with LyCORIS on SDXL can cause NaN loss — this is a general LyCORIS issue, not specific to T-LoRA. See the [LyCORIS documentation](../LYCORIS.md#potential-problems) for details.
