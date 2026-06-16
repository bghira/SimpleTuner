# Prompt2Effect

Prompt2Effect is an experimental CLI-only workflow for training a hypernetwork that generates PEFT LoRA weights from an effect prompt. It is separate from SimpleTuner's normal image/video denoising trainer.

The important distinction is that Prompt2Effect does not make hypernetwork training take 3.3 seconds. It moves the expensive work into a one-time training stage over a library of existing effect LoRAs. After that hypernetwork exists, generating a new LoRA from a prompt is a single forward pass.

## What It Trains

The training samples are existing LoRA checkpoints, not media files:

- an effect prompt
- a PEFT LoRA checkpoint for that effect
- a fixed base model and fixed target layer schema

The prepare step converts each LoRA update into SVD-canonicalized factors. The training loss is normalized MSE over those canonical LoRA factors, not a diffusion loss over latents.

## Supported Families

The scripts currently support:

- `ltxvideo2`
- `wan` I2V flavours
- `hunyuanvideo`

The generated artifact is a normal `pytorch_lora_weights.safetensors` file with PEFT `lora_A`, `lora_B`, and `alpha` keys.

## Files

Prompt2Effect lives under `scripts/prompt2effect/`:

- `prepare.py`: validates a LoRA manifest and writes SVD-canonical targets.
- `train.py`: trains the Prompt2Effect hypernetwork.
- `generate.py`: emits a PEFT LoRA from a trained hypernetwork and an effect prompt.

This is not exposed in the WebUI.

## Manifest

Create a JSONL file with one effect LoRA per line:

```json
{"id":"blue_mood","effect_prompt":"blue mood cinematic atmosphere","lora_path":"/path/to/pytorch_lora_weights.safetensors"}
```

All LoRAs in one Prompt2Effect run must use the same target module schema and the same input/output dimensions. Use `--rank` during prepare to choose the canonical/generated LoRA rank; if omitted, the first LoRA rank is used.

## Prepare Targets

```bash
.venv/bin/python scripts/prompt2effect/prepare.py \
  --manifest /path/to/effects.jsonl \
  --output_dir cache/prompt2effect/wan-i2v-targets \
  --model_family wan \
  --model_flavour i2v-14b-2.1
```

Useful options:

- `--model_family`: `ltxvideo2`, `wan`, or `hunyuanvideo`.
- `--base_model`: override the base model repo or local path.
- `--model_flavour`: use a known family default when `--base_model` is not supplied.
- `--target_modules`: comma-separated PEFT target suffixes, `default`, or `all-linear`.
- `--rank`: generated LoRA rank. Defaults to the first source LoRA rank.
- `--component_subfolder`: base model component subfolder. Defaults to the family transformer subfolder.

`prepare.py` writes:

- `schema.json`
- `targets.safetensors`

It fails if a LoRA is missing required modules, has unexpected modules, or does not match the base model tensor shapes.

## Train

```bash
.venv/bin/python scripts/prompt2effect/train.py \
  --prepared_dir cache/prompt2effect/wan-i2v-targets \
  --output_dir output/prompt2effect/wan-i2v \
  --text_encoder_model google/t5-v1_1-base \
  --max_train_steps 10000
```

The text encoder is frozen and only encodes effect prompts. The base model weights are also frozen and are used as structural conditioning for the hypernetwork.

By default, base weights stay on CPU. Use `--base_weights_device training` only when the selected target layers fit on the accelerator.

## Generate A LoRA

```bash
.venv/bin/python scripts/prompt2effect/generate.py \
  --checkpoint output/prompt2effect/wan-i2v/prompt2effect_hypernetwork.pt \
  --prompt "blue mood cinematic atmosphere" \
  --output output/blue_mood_prompt2effect
```

The output directory will contain `pytorch_lora_weights.safetensors`. Load it like any other SimpleTuner/Diffusers PEFT LoRA.

## Limits

- PEFT linear LoRA only. LyCORIS, convolution LoRA, DoRA magnitude vectors, and arbitrary sidecar tensors are not supported by this workflow yet.
- A hypernetwork is tied to one model family, base model shape, target module schema, and rank.
- The scripts are not integrated with Accelerate, the WebUI, or SimpleTuner's main checkpoint manager.
- Training quality depends on the number and diversity of source effect LoRAs. A handful of LoRAs is enough to test the path, not enough to expect generalization.
- Generated LoRAs should be validated normally before publishing or using in production workflows.
