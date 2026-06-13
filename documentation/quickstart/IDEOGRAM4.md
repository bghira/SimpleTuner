# Ideogram 4 Quickstart

This guide covers LoRA training for Ideogram 4 in SimpleTuner. Ideogram 4 is a large flow-matching image model with strong typography and prompt-following behaviour. The public Ideogram 4 checkpoint is distributed as FP8 weights; SimpleTuner uses that FP8 release by default.

The included starter config is:

```bash
simpletuner/examples/ideogram-fp8.peft-lora/config.json
```

## Hardware requirements

Ideogram 4 is around 9B parameters, so treat it as a large transformer model when planning memory.

Recommended starting points:

- **Best default:** FP8 base weights, bf16 trainable LoRA weights, rank 16-32.
- **Low VRAM fallback:** NF4 quantisation for the base model.
- **High VRAM / fastest iteration:** bf16-upcast transformer weights if you have enough VRAM and want to avoid quantised base loading.

Expected memory varies with rank, optimiser, resolution, validation, and offload strategy. Measured on an H100 80GB with native FP8 (`base_model_precision=fp8-torchao`, `quantize_via=pipeline`), rank 32 LoRA, bf16 mixed precision, gradient checkpointing enabled, 1024px square training, and validation disabled:

| Batch size | Peak VRAM |
| --- | ---: |
| 1 | 15,999 MiB / 15.6 GiB |
| 2 | 20,095 MiB / 19.6 GiB |
| 4 | 28,603 MiB / 27.9 GiB |

Validation has a separate generation peak, so leave extra headroom when `ideogram_validation=true`. On smaller cards, start with FP8 or NF4, rank 8-16, gradient checkpointing, and offload.

Apple GPUs are not recommended for Ideogram 4 training.

### Memory offloading

Grouped module offloading can reduce VRAM pressure when the transformer weights are the bottleneck:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream
```

Optional disk offload:

```bash
--group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Streams are only effective on CUDA; SimpleTuner disables them on ROCm, MPS, and CPU backends.
- Do not combine group offload with other CPU offload strategies.
- Group offload is not compatible with Quanto quantisation.
- Prefer fast local NVMe when offloading to disk.

### Torch compile

For `torch.compile`, prefer regional compilation with native FP8 weights:

```json
{
  "dynamo_backend": "inductor",
  "dynamo_use_regional_compilation": true
}
```

Plain `dynamo_backend="inductor"` also works, but the whole-model first-step compile is slow. Avoid `dynamo_mode="reduce-overhead"` and `dynamo_fullgraph=true` for Ideogram 4 LoRA for now; PEFT LoRA layers can trip CUDA graph output reuse during the second compiled invocation.

## Installation

Install SimpleTuner via pip:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

For manual installation or development setup, see the [installation documentation](../INSTALL.md).

## Configuration

Copy the example config and dataloader into your config directory:

```bash
mkdir -p config/examples
cp simpletuner/examples/ideogram-fp8.peft-lora/config.json config/config.json
cp simpletuner/examples/multidatabackend-ideogram-dreambooth-1024px.json config/examples/multidatabackend-ideogram-dreambooth-1024px.json
```

Important fields:

```json
{
  "model_type": "lora",
  "model_family": "ideogram",
  "model_flavour": "fp8",
  "base_model_precision": "no_change",
  "quantize_via": "cpu",
  "mixed_precision": "bf16",
  "train_batch_size": 1,
  "resolution": 1024,
  "resolution_type": "pixel_area",
  "gradient_checkpointing": true,
  "ideogram_auto_json": true,
  "ideogram_validation": true,
  "ideogram_schedule_mu": 0.0,
  "ideogram_schedule_std": 1.5,
  "validation_guidance": 3.0,
  "validation_num_inference_steps": 12
}
```

### Quantisation choice

Use the FP8 checkpoint first:

```json
{
  "model_flavour": "fp8",
  "base_model_precision": "no_change",
  "quantize_via": "cpu"
}
```

For lower VRAM systems, NF4 is the next recommendation:

```json
{
  "base_model_precision": "nf4-bnb",
  "base_model_default_dtype": "bf16",
  "quantize_via": "cpu"
}
```

If startup is slow or memory-constrained, keep `quantize_via` on `cpu` so the model is prepared before moving to the GPU.

### Text embed cache

Ideogram 4's text encoder output is a concatenation of 13 Qwen hidden-state layers. By default, SimpleTuner projects those raw features through the transformer's frozen `llm_cond_norm` and `llm_cond_proj` layers before writing text embed cache files. This keeps cache files much smaller while preserving the conditioning tensor the transformer consumes.

The projection layers are frozen for both LoRA and full transformer training. For text encoder training, non-standard LoRA, or LoRA targets that explicitly include `llm_cond_norm` or `llm_cond_proj`, SimpleTuner keeps the raw text encoder output in the cache.

The large cache cost comes from feature width, not saved padding. Text embed precomputation writes one file per prompt at that prompt's actual token length; batch padding happens later in memory. The raw 13-layer tensor is `13 * 4096 = 53,248` float32 values per token, or about 0.203 MiB per token before serialization overhead. A 512-token caption is therefore about 104 MiB raw, while the projected bf16 cache is about 4.5 MiB.

If you adapt this path to train a comparable Ideogram-style model from scratch and the text projection is not a fixed pretrained component, disable the projected cache and budget for the much larger raw text embed storage.

Use the full cache only when you intentionally need raw text encoder features or are debugging cache compatibility:

```json
{
  "text_embed_full_cache": true
}
```

This disables the Ideogram 4 projected text embed cache optimisation and stores the full 13-layer text encoder output.

### Validation

Ideogram validation is disabled unless you opt in:

```json
{
  "ideogram_validation": true
}
```

This is temporary. Ideogram's upstream inference path expects an unconditional transformer for CFG, while SimpleTuner currently trains only the conditional transformer by default. With `ideogram_validation=true`, validation uses the conditional transformer for the negative/unconditional pass so you can still check prompt and negative-prompt behaviour.

Use JSON-style validation prompts whenever possible. Short natural-language prompts can trigger Ideogram's built-in filtering or weak prompt behaviour, while structured prompts are more reliable.

## Caption format

Ideogram 4 expects structured JSON captions. A good caption has:

- `high_level_description`
- `style_description`
- `style_description.color_palette` with hex colours
- `compositional_deconstruction.background`
- `compositional_deconstruction.elements`
- optional element `bbox` values as `[x1, y1, x2, y2]`

Example:

```json
{
  "high_level_description": "A cinematic 35mm film photograph of a lone wooden sailboat on a glassy lake at sunset, the boat on a right-third vertical with the horizon at the lower third, in a cool muted blue palette.",
  "style_description": {
    "aesthetics": "Cinematic, minimal, serene, quiet stillness.",
    "lighting": "Cool overcast dusk light with a small warm sun low at the horizon; muted and low-contrast.",
    "photo": "35mm motion-picture film still, 16:9 framing, subtle grain, slightly desaturated.",
    "medium": "Photograph.",
    "color_palette": ["#1B3A5C", "#5B8FB9"]
  },
  "compositional_deconstruction": {
    "background": "Windless evening on a wide lake; horizon at the lower third. Dusty blue-violet sky with a small amber sun at the horizon and a thin gold streak across the glassy teal water. Subtle 35mm grain.",
    "elements": [
      {
        "type": "obj",
        "bbox": [380, 590, 660, 720],
        "desc": "Lone wooden sailboat on the right-third vertical in the midground, dark varnished hull and a single tall mast with a slack white sail hanging limp in the still air."
      }
    ]
  }
}
```

### Auto JSON wrapping

If your dataset has mixed plain text and JSON captions, keep this enabled:

```json
{
  "ideogram_auto_json": true
}
```

Plain prompts are wrapped into the Ideogram JSON schema. Existing JSON captions are canonicalised and preserved. This is useful when converting older datasets, but hand-written JSON captions are still better.

### Prompt upsampling

Prompt upsampling can be enabled with:

```json
{
  "ideogram_prompt_upsample": true,
  "ideogram_prompt_enhancer_head_id": "diffusers/qwen3-vl-8b-instruct-lm-head"
}
```

This rewrites validation/training prompts through the Ideogram prompt upsampler before JSON conversion. It is optional and slower. Leave it disabled until the base training path is working.

## Dataset configuration

The included example uses the Domokun demo dataset and a JSON instance prompt:

```bash
simpletuner/examples/multidatabackend-ideogram-dreambooth-1024px.json
```

For your own local dataset, use the same shape:

```json
[
  {
    "id": "dreambooth-1024",
    "type": "local",
    "crop": true,
    "crop_style": "random",
    "crop_aspect": "square",
    "minimum_image_size": 256,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "metadata_backend": "discovery",
    "caption_strategy": "instanceprompt",
    "instance_data_dir": "datasets/ideogram-subject",
    "instance_prompt": "{\"high_level_description\":\"A detailed photograph of <token> as the main subject.\",\"style_description\":{\"aesthetics\":\"Clean, detailed, natural.\",\"lighting\":\"Soft natural light.\",\"photo\":\"Sharp 35mm photograph, square framing.\",\"medium\":\"Photograph.\",\"color_palette\":[\"#4A4A4A\",\"#D8D0C4\"]},\"compositional_deconstruction\":{\"background\":\"Simple uncluttered background.\",\"elements\":[{\"type\":\"obj\",\"bbox\":[260,180,760,860],\"desc\":\"<token> centered as the main subject.\"}]}}",
    "cache_dir_vae": "cache/vae/ideogram/dreambooth-1024"
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/ideogram"
  }
]
```

Text embeds are cached without padding; padding is applied at batch time. This keeps cached prompts compact while still allowing variable-length structured captions.

## LoRA targets

The standard PEFT example targets attention projections:

```json
{
  "lora_type": "standard",
  "lora_rank": 32
}
```

For LyCORIS/LoKr, the default SimpleTuner targets work because Ideogram exposes `Attention` and `FeedForward` module classes:

```json
{
  "lora_type": "lycoris",
  "lycoris_config": "config/lycoris_config.json"
}
```

Full-matrix LoKr can be very large. In one smoke run, a one-step full-matrix LoKr adapter saved as a multi-GB file. Start with PEFT LoRA unless you specifically need LyCORIS.

## Training

From the SimpleTuner directory:

```bash
simpletuner train
```

or, from a development checkout:

```bash
CONFIG_BACKEND=json CONFIG_PATH=config/config.json .venv/bin/python simpletuner/train.py
```

This will cache text embeds and VAE latents, then begin training.

## Loss expectations

Ideogram loss can look high compared with other models. Values near or above `1.0` do not automatically mean the model is broken or that validation images will be incoherent.

In test runs, Ideogram produced coherent validation images even while step losses bounced through roughly `0.3-1.3`, with occasional higher spikes. Judge the run by validation image coherence, prompt adherence, and whether loss is exploding, not by expecting a very low scalar loss.

## Troubleshooting

- **Validation is skipped:** add `--ideogram_validation` or set `"ideogram_validation": true`.
- **Validation is noise or a flat colour:** ensure validation autocast is disabled for Ideogram, use the current SimpleTuner pipeline, and prefer JSON validation prompts.
- **Short prompts produce weak or filtered images:** use structured JSON captions or keep `ideogram_auto_json=true`.
- **OOM at startup:** use native FP8 first, then NF4 for lower VRAM, lower LoRA rank, enable gradient checkpointing, and use CPU quantisation/offload.
- **LoKr attaches zero modules:** use current SimpleTuner where Ideogram modules are named `Attention` and `FeedForward`, or target those names explicitly.
- **Adapter files are huge:** full-matrix LoKr is expected to be much larger than PEFT LoRA. Use standard LoRA for quick iteration.
