# Prompt2Effect Scripts

This folder contains a CLI-only Prompt2Effect workflow for training a LoRA-generating
hypernetwork from existing PEFT LoRA checkpoints.

Supported model families:

- `ltxvideo2`
- `wan` I2V flavours
- `hunyuanvideo`

The workflow is intentionally separate from SimpleTuner's normal media-sample
training loop and WebUI. Prompt2Effect samples are effect prompts paired with
existing LoRA files, and the loss is normalized MSE over SVD-canonicalized LoRA
factors.

## Manifest

Use JSONL with one LoRA per line:

```json
{"id":"blue_mood","effect_prompt":"blue mood cinematic atmosphere","lora_path":"/path/to/pytorch_lora_weights.safetensors"}
```

All LoRAs in a run must use the same PEFT target module schema and input/output
dimensions. Use `--rank` to choose the canonical/generated LoRA rank; when it is
omitted, the first LoRA rank is used.

## Prepare Targets

```bash
.venv/bin/python scripts/prompt2effect/prepare.py \
  --manifest /path/to/effects.jsonl \
  --output_dir cache/prompt2effect/wan-i2v-targets \
  --model_family wan \
  --model_flavour i2v-14b-2.1
```

`prepare.py` validates the base model tensors, verifies every LoRA has the same
schema, applies PEFT alpha scaling, and stores SVD-canonical targets.

## Train

```bash
.venv/bin/python scripts/prompt2effect/train.py \
  --prepared_dir cache/prompt2effect/wan-i2v-targets \
  --output_dir output/prompt2effect/wan-i2v \
  --text_encoder_model google/t5-v1_1-base \
  --max_train_steps 10000
```

Use `--base_weights_device training` only when the selected target layers fit on
the accelerator. The default keeps base weights on CPU and moves each layer as
needed.

## Generate LoRA

```bash
.venv/bin/python scripts/prompt2effect/generate.py \
  --checkpoint output/prompt2effect/wan-i2v/prompt2effect_hypernetwork.pt \
  --prompt "blue mood cinematic atmosphere" \
  --output output/blue_mood_prompt2effect
```

The output is a standard `pytorch_lora_weights.safetensors` file with PEFT
`lora_A`, `lora_B`, and `alpha` keys.
