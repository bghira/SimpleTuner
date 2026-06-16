# Prompt2Effect

Prompt2Effect ek experimental CLI-only workflow hai jo effect prompt se PEFT LoRA weights generate karne wali hypernetwork train karta hai. Yeh SimpleTuner ke normal image/video denoising trainer se alag hai.

Important baat yeh hai ki Prompt2Effect hypernetwork training ko 3.3 seconds ka nahi banata. Mehenga kaam ek one-time training stage me chala jata hai, jahan existing effect LoRAs ki library use hoti hai. Us hypernetwork ke train ho jane ke baad, naye prompt se LoRA generate karna ek single forward pass hota hai.

## Kya Train Hota Hai

Training samples media files nahi, existing LoRA checkpoints hote hain:

- ek effect prompt
- us effect ke liye PEFT LoRA checkpoint
- ek fixed base model aur fixed target layer schema

Prepare step har LoRA update ko SVD-canonicalized factors me convert karta hai. Training loss in canonical LoRA factors par normalized MSE hota hai, latents par diffusion loss nahi.

## Supported Families

Scripts abhi in families ko support karte hain:

- `ltxvideo2`
- `wan` I2V flavours
- `hunyuanvideo`

Generated artifact ek normal `pytorch_lora_weights.safetensors` file hoti hai jisme PEFT `lora_A`, `lora_B`, aur `alpha` keys hoti hain.

## Files

Prompt2Effect `scripts/prompt2effect/` me hai:

- `prepare.py`: LoRA manifest validate karta hai aur SVD-canonical targets likhta hai.
- `train.py`: Prompt2Effect hypernetwork train karta hai.
- `generate.py`: trained hypernetwork aur effect prompt se PEFT LoRA emit karta hai.

Yeh WebUI me exposed nahi hai.

## Manifest

Ek JSONL file banao jisme har line par ek effect LoRA ho:

```json
{"id":"blue_mood","effect_prompt":"blue mood cinematic atmosphere","lora_path":"/path/to/pytorch_lora_weights.safetensors"}
```

Ek Prompt2Effect run ke sabhi LoRAs ko same target module schema aur same input/output dimensions use karne honge. Prepare ke waqt `--rank` se canonical/generated LoRA rank choose karo; agar omit kiya gaya, pehle LoRA ka rank use hoga.

## Targets Prepare Karna

```bash
.venv/bin/python scripts/prompt2effect/prepare.py \
  --manifest /path/to/effects.jsonl \
  --output_dir cache/prompt2effect/wan-i2v-targets \
  --model_family wan \
  --model_flavour i2v-14b-2.1
```

Useful options:

- `--model_family`: `ltxvideo2`, `wan`, ya `hunyuanvideo`.
- `--base_model`: base model repo ya local path override karta hai.
- `--model_flavour`: jab `--base_model` na diya ho tab known family default use karta hai.
- `--target_modules`: comma-separated PEFT target suffixes, `default`, ya `all-linear`.
- `--rank`: generated LoRA rank. Default pehle source LoRA ka rank hai.
- `--component_subfolder`: base model component subfolder. Default family transformer subfolder hai.

`prepare.py` likhta hai:

- `schema.json`
- `targets.safetensors`

Yeh fail karega agar kisi LoRA me required modules missing hon, unexpected modules hon, ya base model tensor shapes match na karein.

## Train

```bash
.venv/bin/python scripts/prompt2effect/train.py \
  --prepared_dir cache/prompt2effect/wan-i2v-targets \
  --output_dir output/prompt2effect/wan-i2v \
  --text_encoder_model google/t5-v1_1-base \
  --max_train_steps 10000
```

Text encoder frozen hota hai aur sirf effect prompts encode karta hai. Base model weights bhi frozen hote hain aur hypernetwork ke structural conditioning ke roop me use hote hain.

Default me base weights CPU par rehte hain. `--base_weights_device training` sirf tab use karo jab selected target layers accelerator par fit hoti hon.

## LoRA Generate Karna

```bash
.venv/bin/python scripts/prompt2effect/generate.py \
  --checkpoint output/prompt2effect/wan-i2v/prompt2effect_hypernetwork.pt \
  --prompt "blue mood cinematic atmosphere" \
  --output output/blue_mood_prompt2effect
```

Output directory me `pytorch_lora_weights.safetensors` hoga. Use kisi bhi SimpleTuner/Diffusers PEFT LoRA ki tarah load karo.

## Limits

- Sirf PEFT linear LoRA. LyCORIS, convolution LoRA, DoRA magnitude vectors, aur arbitrary sidecar tensors is workflow me abhi supported nahi hain.
- Ek hypernetwork ek model family, base model shape, target module schema, aur rank se tied hota hai.
- Scripts Accelerate, WebUI, ya SimpleTuner ke main checkpoint manager se integrated nahi hain.
- Training quality source effect LoRAs ki quantity aur diversity par depend karti hai. Kuch LoRAs path test karne ke liye kaafi hain, generalization expect karne ke liye nahi.
- Generated LoRAs ko publish karne ya production workflows me use karne se pehle normal tarike se validate karna chahiye.
