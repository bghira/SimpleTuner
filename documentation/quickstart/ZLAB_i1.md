# zlab i1 Quickstart

In this example, we'll be training a LoRA for the [zlab-princeton i1](https://huggingface.co/zlab-princeton/i1-3B) image model. i1 is a 3B flow-matching transformer released with a JAX/TPU training recipe and PyTorch inference weights. SimpleTuner trains it through a native PyTorch integration and uses a Diffusers safetensors conversion at [`bghira/zlab-i1-diffusers`](https://huggingface.co/bghira/zlab-i1-diffusers).

The important bit: i1 is not a Flux clone. It uses the FLUX.2 VAE, a T5Gemma text encoder, 32-channel latents, and a learned null caption for classifier-free guidance.

## Hardware requirements

i1 is much smaller than Flux.2, but the text encoder and VAE still make startup fairly heavy.

For 1024px LoRA training, expect roughly:

- a modern 24G GPU with int8 quantisation for small LoRA runs
- more comfortable training on 40G+ cards
- multi-GPU systems for larger ranks, bigger datasets, or less quantisation

The example configs use:

- `base_model_precision`: `int8-quanto`
- `mixed_precision`: `bf16`
- `gradient_checkpointing`: `true`
- `train_batch_size`: `1`

Apple GPUs are not recommended for i1 training. CUDA is the expected path.

## Available examples

Two ready-to-run examples are included:

```bash
simpletuner train example=zlab-i1.peft-lora
simpletuner train example=zlab-i1.lycoris-lokr
```

The PEFT example is the safer first run. The LyCORIS LoKr example is useful when you specifically want LoKr factorisation instead of a standard LoRA adapter.

## Configuration notes

If you are adapting an existing config, the core settings are:

```json
{
  "model_type": "lora",
  "model_family": "zlab_i1",
  "model_flavour": "3b",
  "pretrained_vae_model_name_or_path": "black-forest-labs/FLUX.2-dev",
  "base_model_precision": "int8-quanto",
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "validation_resolution": "1024x1024",
  "validation_guidance": 12.0,
  "validation_guidance_rescale": 0.7,
  "validation_num_inference_steps": 250
}
```

The `3b` flavour resolves to `bghira/zlab-i1-diffusers`, where the transformer is stored in the standard Diffusers `transformer/` subfolder as safetensors. You only need to set `pretrained_transformer_model_name_or_path` when testing a custom conversion.

## Validation

SimpleTuner validation works through the native i1 pipeline. The defaults follow upstream inference:

- 1024x1024 output
- 250 denoising steps
- CFG guidance around `12`
- CFG rescale around `0.7`

For quick smoke tests, you can temporarily reduce validation steps:

```bash
simpletuner train example=zlab-i1.peft-lora validation_num_inference_steps=4 num_eval_images=1
```

Four steps is only a pipeline sanity check. Use the full 250-step setting before judging quality.

## Advanced features

i1 participates in the common SimpleTuner transformer feature paths:

- TwinFlow works in native flow-matching mode. The i1 timestep input is intentionally ignored by the upstream model, so TwinFlow changes the noisy latent trajectory and target construction rather than adding a new time embedding.
- CREPA Self-Flow and LayerSync use the i1 image-token hidden-state buffer. Set CREPA block indices against the 29 i1 transformer layers.
- TREAD routes image tokens only. Text tokens stay intact so the T5Gemma conditioning mask keeps its original semantics.
- Validation accepts CFG Zero*, CFG step skip through `validation_no_cfg_until_timestep`, and skip-layer guidance through `validation_guidance_skip_layers`.
- RamTorch, Musubi block swap, and VAE tiling are supported. Keep RamTorch and Musubi mutually exclusive.

## Dataset notes

i1 uses the FLUX.2 VAE and expects 32-channel latents. Do not reuse a cache created for SDXL, Flux.1, PixArt, or another model family.

For a local DreamBooth-style dataset, keep the usual SimpleTuner layout:

```json
[
  {
    "id": "my-i1-dataset",
    "type": "local",
    "instance_data_dir": "/datasets/my-subject",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/zlab_i1/my-i1-dataset"
  }
]
```

Short prompts can work, but i1 responds better when validation prompts describe the composition clearly. If the model seems to ignore a small detail, try rewriting the prompt before assuming training failed.

## LoRA targets

The default PEFT target list covers i1 attention projections, feed-forward layers, and linear layers:

```json
[
  "qkv_image",
  "qkv_text",
  "proj_image",
  "proj_text",
  "w12",
  "w3",
  "linear"
]
```

For LyCORIS LoKr, the example targets the i1 block classes:

```json
{
  "target_module": ["MMDiTAttention", "SwiGLUFFN"]
}
```

## Practical starting point

Start with the PEFT example unchanged and confirm that:

1. The base benchmark image is created.
2. Training produces finite loss.
3. A validation image appears after checkpoint validation.
4. `pytorch_lora_weights.safetensors` is saved in the output directory.

Once that works, change the dataset and prompts. Keep the default validation settings until you have a baseline image you trust.
