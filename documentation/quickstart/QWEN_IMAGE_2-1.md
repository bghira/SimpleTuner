# Qwen Image 2.1 quickstart

[Qwen Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) uses a 7B, 32-block image transformer, a Qwen3-VL text encoder and a 64-channel VAE with 16× spatial compression. SimpleTuner selects it by default with `model_family: "qwen_image"` and `model_flavour: "v2.1"`. This guide covers text-to-image LoRA training.

For the 20B `v1.0` / `v2.0` models, use the [older Qwen Image guide](QWEN_IMAGE.md). Paired-reference training for older `edit-*` flavours is covered in [Qwen Edit](QWEN_EDIT.md). Their adapters, text embeddings and latent caches are not interchangeable with 2.1.

[Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)

## Install

Use Python 3.12–3.14 and follow the [installation guide](../INSTALL.md) for your platform. The presets below were tested on NVIDIA GPUs. `webshart` is included in SimpleTuner’s dependencies; the regularisation dataset requires network access and local cache space.

```bash
pip install 'simpletuner[cuda]'
```

<a id="vram-presets"></a>

## Choose a VRAM preset

The standard examples combine [assistant v3](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v3), synthetic regularisation, REPA and automatic flow schedule shifting. The default `qwen_image.peft-lora` uses the same 512px + 1024px recipe as the 48 GB preset. The 24/32 GB presets use only 512px, including validation.

| VRAM budget | Example | Base resolutions | Updates | Gradient checkpointing interval |
| --- | --- | --- | --- | --- |
| 24 GB | `qwen_image-2.1-24g.peft-lora` | 512px | 2000 | 1 |
| 32 GB | `qwen_image-2.1-32g.peft-lora` | 512px | 2000 | 2 |
| 48 GB | `qwen_image-2.1-48g.peft-lora` | 512px + 1024px | 4000 | 2 |
| 80 GB | `qwen_image-2.1-80g.peft-lora` | 512px + 1024px | 4000 | 2 |
| 144 GB | `qwen_image-2.1-144g.peft-lora` | 512px + 1024px | 4000 | 2 |

```bash
simpletuner train example=qwen_image-2.1-48g.peft-lora
```

All presets use BF16, rank/alpha 32, batch 1, AdamW BF16 at `1e-4`, 25 warmup updates and norm clipping at 1.0. REPA uses `dinov2_vitg14`, block 8, weight 0.5, encoder size 518 and spatial alignment; temporal distance is 0. Auto shift is enabled with static shift 0. The assistant is frozen during training and disabled for validation. Regularisation matches the bare base prediction, with both adapters disabled for the parent target.

The normal probabilistic sampler gives half the total weight to `RareConcepts/Domokun` with trigger `🟫`, and half to `webshart/qwen-image-2.1-generated-images` with `is_regularisation_data: true`. Multi-scale presets divide each half equally between 512px and 1024px. Area-based aspect buckets preserve image proportions: 0.262144 and 1.048576 megapixels. Synthetic square, portrait and landscape backends share their resolution's regularisation weight equally. Sampling is probabilistic, not strict alternation. Validation and checkpoints run every 250 updates.

The bundled dataloaders cap each synthetic aspect subset at 1,024 images and use separate latent caches for each resolution and source. VAE tiling is disabled. These update budgets are starting points: inspect checkpoint images before extending a run.

<details markdown="1">
<summary>Measured memory checks</summary>

L40S memory smoke tests used 16 updates, four images per backend, validation and checkpoint saving. The 512px interval-1 recipe passed a 24 GiB allocation limit, peaking at 20.06 GiB allocated / 21.10 GiB reserved by PyTorch. The multi-scale interval-2 recipe passed on L40S at 32.49 / 41.21 GiB, but ran out of memory under a 32 GiB limit. These are bounded-subset memory checks, not throughput or convergence measurements; the lower budgets were simulated on L40S rather than tested on separate cards. The final 32 GB preset, using 512px and interval 2, also passed: 22.12 GiB allocated / 23.68 GiB reserved.

</details>

## Use your own data

Copy the selected example’s `config.json` and `dataloader.json` into your training environment, and supply your own prompt library. Update `data_backend_config`, `user_prompt_library` and `output_dir` to match those files. See the [training tutorial](../TUTORIAL.md) and [dataloader reference](../DATALOADER.md) for configuration layout.

Replace the Domokun backends with your concept or photo dataset. In a multi-scale run, define separate 512px and 1024px backends, with different IDs and VAE cache paths. With `resolution_type: "area"`, use **0.262144** and **1.048576** megapixels; `crop: false` preserves aspect ratios. Keep `aspect_bucket_alignment: 32` in the training config.

For the combined recipe, retain the synthetic backends with `is_regularisation_data: true` and mark your training backends false. Keep half the sampling weight for training data and half for regularisation; split each half across resolutions. Selection uses the probabilistic sampler, not strict alternation. Repeats control dataset availability, not the sampling probability. Start fresh if dataset, resolution, batching or distributed topology changes.

## Choose the training assistant

Main and the bundled examples currently use **assistant v3**. It is frozen at strength 1 during training and disabled for validation. On regularisation batches, the parent target is the **bare base prediction**, with both the trainable LoRA and assistant disabled.

[Assistant v3](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v3) is the default rank-64 adapter, trained for 30k updates across 512/1024/1536/2048 resolutions. The recorded preset memory checks used v2; check available VRAM when using v3. To reproduce older runs, explicitly select [assistant v2](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v2) with the following override:

```json
{
  "assistant_lora_path": "SimpleTuner/Qwen-Image-2.1-training-assistant-v2",
  "assistant_lora_strength": 1.0,
  "assistant_lora_inference_strength": 0.0
}
```

Set `disable_assistant_lora: true` to train without an assistant. Use the same setting when creating a new assistant. For downstream inference, load only your trained LoRA; the assistant and REPA projector are training components.

## Validate learning and quality

The examples validate and save every 250 updates, using 40 inference steps, true CFG 1 and seed 42. They validate at 512px on 24/32 GB presets, and at both 512px and 1024px on the larger presets. These are the relevant settings to retain in a custom config:

```json
{
  "validation_guidance": 1.0,
  "validation_guidance_real": 1.0,
  "validation_num_inference_steps": 40,
  "validation_seed": 42,
  "validation_step_interval": 250,
  "checkpoint_step_interval": 250
}
```

Include your trigger in concept prompts and include unrelated subjects to inspect coherence and image quality. Keep prompt, seed, decoder, resolution and guidance matched when comparing checkpoints. Training and output resolution are separate: test a 512px-trained adapter at 1024px too. These recipes do not require 2048px concept data.

The [experiments book](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-LoRA-experiments) includes assistant, regularisation, REPA, auto-shift and longer-training comparisons. Some runs regress and later recover, so one weak checkpoint is not enough to judge the trajectory. The [photo-aesthetics v3 comparison](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-LoRA-photo-aesthetics-v3) shows denser scene detail from 50k updates at 512px than from 10k at 1024px, even at 1MP output; those runs have different update budgets.

## VAE and inference

Qwen Image 2.1 uses [Ollin’s texture-fixed VAE](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1) by default for validation and other VAE decoding. Only the decoder was fine-tuned; the encoder is unchanged, so existing 2.1 training latents remain compatible. The VAE revision is pinned independently of the base model. To reproduce outputs with the original VAE, set `pretrained_vae_model_name_or_path: "Qwen/Qwen-Image-2.1"`. Explicit VAE overrides and older flavours keep their existing behaviour.

For standard PEFT adapters, the vendored SimpleTuner pipeline avoids requiring a Git installation of Diffusers. This example reads your LoRA export and excludes training-only REPA projector tensors. It uses full-frame decoding and no assistant:

Model CPU offload keeps the text encoder, transformer and VAE from occupying GPU memory together. With sufficient VRAM, replace `pipe.enable_model_cpu_offload()` with `pipe.to("cuda")` for faster repeated inference.

```python
import torch
from safetensors.torch import load_file
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from simpletuner.helpers.models.qwen_image.autoencoder_21 import AutoencoderKLQwenImage21
from simpletuner.helpers.models.qwen_image.pipeline_21 import QwenImage21Pipeline
from simpletuner.helpers.models.qwen_image.transformer_21 import QwenImage21Transformer2DModel

vae = AutoencoderKLQwenImage21.from_pretrained(
    "madebyollin/texture-fix-vae-for-qwen-image-2.1",
    revision="e9f84623d22c47f8bc9fb799bc54201fa53cf80b",
    torch_dtype=torch.bfloat16,
)
model_id = "Qwen/Qwen-Image-2.1"
pipe = QwenImage21Pipeline(
    transformer=QwenImage21Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
    ),
    vae=vae,
    text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, subfolder="text_encoder", dtype=torch.bfloat16,
    ),
    processor=Qwen3VLProcessor.from_pretrained(model_id, subfolder="processor"),
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler"),
)
pipe.enable_model_cpu_offload()
pipe.vae.disable_tiling()
pipe.vae.enable_slicing()
weights = load_file("output/my-qwen21/pytorch_lora_weights.safetensors")
weights = {
    key: value for key, value in weights.items()
    if ".lora_A." in key or ".lora_B." in key
}
pipe.load_lora_weights(weights, adapter_name="trained")
pipe.set_adapters("trained", adapter_weights=1.0)
image = pipe(
    prompt="🟫 wearing a red scarf, standing in a snowy forest",
    width=1024, height=1024, num_inference_steps=40, true_cfg_scale=1.0,
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]
image.save("output.png")
```

## Memory and image artifacts

Start with BF16 (`base_model_precision: "no_change"`). Reduce batch size or select a smaller preset before quantising; quantisation can reduce memory but does not guarantee faster training. Regional compilation is enabled in the examples. Large captions, additional resolutions, validation and REPA can change peak VRAM. Use measured headroom before increasing batch size.

Keep `vae_enable_tiling: false` when memory permits. The texture-fixed decoder addresses the canvas-like texture; tiled colour seams are a separate spatial-context issue. Norm clipping at 1.0 means `grad_clip_method: "norm"` and `max_grad_norm: 1.0`; it is not the elementwise 0.01 clipping used by the old AnyFlow pilot.

<details markdown="1">
<summary>Measured memory checks: VAE</summary>

Qwen Image 2.1 decodes single images without retaining unused temporal feature caches. In an isolated BF16 H200 decode of one 2048×2048 image, this reduced peak allocated memory from 26.87 to 15.28 GiB with identical output. Tiled decoding saves more memory but can introduce colour seams by limiting spatial context; removing the unused caches does not fix those seams.

</details>

## Historical and experimental recipes

The recipes below create assistants or test distillation. They are separate from the recommended downstream recipe above; their older pilot settings and results are retained for reference.

<details markdown="1">
<summary>AnyFlow pilot and optimizer history</summary>

<a id="experimental-anyflow-pilot"></a>

### Experimental AnyFlow pilot

The three `qwen_image-2.1-anyflow-stage*.peft-lora` examples test whether interval distillation can retain the base model's behavior while introducing `🟫`. They are experimental; Qwen's model card does not establish that guidance distillation caused the earlier deterioration.

All stages use 1024px, BF16, AdamW, rank 32 and batch 4. Stage 1 disables gradient checkpointing on H200; stages 2 and 3 checkpoint contiguous two-block groups. This is a separate distillation workload from the assisted REPA presets above.

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

</details>

<details markdown="1">
<summary>Creating an assistant: online, multi-resolution and offline recipes</summary>

<a id="assistant-lora"></a>

### Training an assistant LoRA from captions

`qwen_image-2.1-assistant-lora.peft-lora` is an experimental L40S starting point: BF16, batch 1, interval-2 checkpointing, rank 32 and AdamW BF16 at `1e-4`. It budgets 1,000 updates with validation/checkpoints every 50. Replace the twelve smoke captions with diverse captions before a substantive run; the example is not a validated convergence recipe.

`grad_clip_method: "norm"`, `max_grad_norm: 1.0`.

Set `distillation_method: assistant_lora`. The caption backend precomputes text embeddings, then each batch generates fresh base-model latents with the adapter disabled, 40 native inference steps and CFG 1. The private generation pipeline reuses the transformer without loading a VAE, processor or text encoder. The adapter is restored before ordinary denoising training. Terminal latents are never cached. This currently supports Qwen Image 2.1 text-to-image only.

`distillation_config.assistant_lora` accepts `num_inference_steps` (default 40), `resolutions` (a nonempty list of `[width, height]`, default `[[1024, 1024]]`) and `seed` (default 42). Qwen dimensions must be multiples of 32. Resolutions cycle per batch; noise seeds advance per sample, including short batches. Checkpoints preserve these counters. Resume with unchanged dataset, batch size, accumulation and distributed topology. Text-cache on-demand mode is not supported.

For a plumbing test, use 8 updates and 2 teacher steps. Restore 40 teacher steps before judging samples. Review a larger run at 100, 250, 500 and 1,000 updates; compare adapter-enabled images against the same base-model prompts/seeds. A useful assistant must still be tested in a separate concept-training run.

Qwen Image 2.1 LoRA training now loads [training assistant v3](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v3) by default. It stays frozen during training and is disabled for validation. Set `disable_assistant_lora: true` to opt out, or set `assistant_lora_path` to use another adapter. Older flavours do not receive this default. When training a new assistant, keep `disable_assistant_lora: true` as in the assistant-training examples.

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

</details>
