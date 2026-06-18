# Boogu-Image 0.1 Quickstart

यह guide SimpleTuner में Boogu-Image 0.1 के लिए LoRA और LyCORIS LoKr training बताती है। Boogu-Image एक flow-matching image model है जिसमें text-to-image, turbo, और edit flavours हैं। SimpleTuner integration local pipeline और transformer code इस्तेमाल करता है, और exported pipeline checkpoints Hugging Face के `SimpleTuner` namespace में रखे जाते हैं।

शामिल starter configs:

```bash
simpletuner/examples/boogu-image-v0.1.peft-lora/config.json
simpletuner/examples/boogu-image-v0.1.lycoris-lokr/config.json
```

## Hardware requirements

Boogu-Image को large transformer image model की तरह plan करें। पहली runs के लिए 1024px training, batch size 1, bf16 mixed precision, और gradient checkpointing इस्तेमाल करें।

Recommended starting points:

- **Default:** `v0.1-base`, bf16 LoRA weights, rank 16।
- **Lower VRAM:** `v0.1-base-fp8`, `v0.1-turbo-fp8`, या `v0.1-edit-fp8` जैसे FP8 flavour।
- **Fast validation / inference:** turbo flavour, नीचे दिए assistant LoRA status को ध्यान में रखकर।
- **Editing:** paired conditioning data के साथ `v0.1-edit` या `v0.1-edit-fp8`।

Memory usage rank, optimizer, validation resolution, offload, compile settings, और FP8 weights पर निर्भर करती है। एक single H100 included PEFT LoRA example को 1024px पर 1000 steps तक benchmark और validation samples enabled के साथ train कर सकता है।

छोटी GPUs पर FP8 weights, rank 8-16, `train_batch_size=1`, gradient checkpointing, और model/group offload से शुरू करें।

### Memory offloading

जब transformer weights VRAM bottleneck हों, group offload मदद कर सकता है:

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

- Streams केवल CUDA पर effective हैं; SimpleTuner ROCm, MPS, और CPU पर इन्हें disable करता है।
- Group offload को अन्य CPU offload strategies के साथ combine न करें।
- Disk offload के लिए fast local NVMe prefer करें।

### Torch compile और attention

NVIDIA GPUs पर available होने पर Hugging Face Hub kernel attention aliases इस्तेमाल करें:

```json
{
  "attention_mechanism": "flash-attn-3-hub",
  "dynamo_backend": "inductor",
  "dynamo_use_regional_compilation": true
}
```

अगर किसी GPU/driver stack पर compiled validation black images बनाता है, तो training recipe बदलने से पहले torch compile disable करके retest करें।

## Installation

SimpleTuner को pip से install करें:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

Manual installation या development setup के लिए [installation documentation](../INSTALL.md) देखें।

## Environment setup

### Web interface method

SimpleTuner WebUI Boogu-Image training config बना सकता है:

```bash
simpletuner server
```

http://localhost:8001 खोलें और model family में `boogu_image` चुनें।

### Manual / command-line method

`config/config.json.example` को `config/config.json` में copy करें:

```bash
cp config/config.json.example config/config.json
```

ये values review करें:

- `model_type` - `lora`।
- `lora_type` - PEFT LoRA के लिए `standard`, LyCORIS LoKr के लिए `lycoris`।
- `model_family` - `boogu_image`।
- `model_flavour` - `v0.1-base`, `v0.1-base-fp8`, `v0.1-turbo`, `v0.1-turbo-fp8`, `v0.1-edit`, या `v0.1-edit-fp8`।
- `pretrained_model_name_or_path` - आम तौर पर unset रखें ताकि flavour `SimpleTuner/Boogu-Image-0.1-*` pipeline चुने।
- `output_dir` - checkpoints और validation images का output directory।
- `train_batch_size` - `1` से शुरू करें।
- `resolution` - `1024` से शुरू करें।
- `resolution_type` - multi-aspect buckets के लिए `pixel_area`।
- `validation_resolution` - `1024x1024`; multiple sizes comma-separated हो सकते हैं।
- `validation_guidance` - base/edit के लिए करीब `4.0`।
- `validation_num_inference_steps` - base/edit के लिए करीब `30`; turbo कम steps इस्तेमाल कर सकता है।
- `mixed_precision` - modern NVIDIA GPUs पर `bf16`।
- `gradient_checkpointing` - enabled रखें।
- `flow_schedule_shift` - examples `3` इस्तेमाल करते हैं।

Minimal PEFT LoRA config:

```json
{
  "model_type": "lora",
  "model_family": "boogu_image",
  "model_flavour": "v0.1-base",
  "lora_type": "standard",
  "lora_rank": 16,
  "lora_alpha": 16,
  "output_dir": "output/models-boogu-image-v0.1",
  "train_batch_size": 1,
  "validation_resolution": "1024x1024",
  "validation_guidance": 4.0,
  "validation_num_inference_steps": 30,
  "validation_prompt": "a polished product photo of a ceramic mug on a walnut desk",
  "validation_steps": 50,
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "flow_schedule_shift": 3,
  "optimizer": "adamw_bf16",
  "learning_rate": 1e-4,
  "lr_scheduler": "constant_with_warmup",
  "lr_warmup_steps": 10,
  "max_train_steps": 1000,
  "resolution": 1024,
  "resolution_type": "pixel_area",
  "data_backend_config": "config/examples/multidatabackend-small-dreambooth-1024px.json"
}
```

## Examples run करना

```bash
simpletuner train example=boogu-image-v0.1.peft-lora
simpletuner train example=boogu-image-v0.1.lycoris-lokr
```

Development checkout form:

```bash
simpletuner train env=examples/boogu-image-v0.1.peft-lora
simpletuner train env=examples/boogu-image-v0.1.lycoris-lokr
```

## FP8 flavours

Exported FP8 pipeline weights load करने के लिए `-fp8` flavours इस्तेमाल करें:

```json
{
  "model_family": "boogu_image",
  "model_flavour": "v0.1-base-fp8"
}
```

यही pattern `v0.1-turbo-fp8` और `v0.1-edit-fp8` पर भी लागू होता है। SimpleTuner को Boogu `.bin` files पर point करने की जरूरत नहीं है।

## Turbo assistant LoRA

SimpleTuner `v0.1-turbo` और `v0.1-turbo-fp8` के लिए assistant LoRA code path enable करता है। Adapter path अभी `None` placeholder है क्योंकि इस integration के लिए अलग assistant adapter अभी publish नहीं हुआ है।

जब तक adapter available नहीं है, turbo को exported pipeline target की तरह इस्तेमाल करें और quality को validation से directly check करें। सबसे predictable baseline के लिए `v0.1-base` से शुरू करें।

## Edit training

Boogu edit flavours को paired conditioning data चाहिए। [Qwen Image Edit quickstart](./QWEN_EDIT.md) में बताई paired-reference dataset structure इस्तेमाल करें।

Text-to-image LoRA runs के लिए base या turbo flavours इस्तेमाल करें।

## Validation prompts

`validation_prompt` primary validation prompt है। अधिक coverage के लिए prompt library जोड़ें:

```json
{
  "product": "a polished product photo of <token> on a walnut desk",
  "studio": "a clean studio portrait of <token> with softbox lighting",
  "cinematic": "a cinematic scene featuring <token>, detailed lighting, shallow depth of field"
}
```

Config में इसे point करें:

```json
{
  "validation_prompt_library": "config/user_prompt_library.json"
}
```

Overfitting, prompt collapse, और style drift पकड़ने के लिए sufficiently distinct prompts इस्तेमाल करें।

## Inference

Training के बाद saved adapter को उसी Boogu-Image pipeline flavour के साथ load करें जो training में इस्तेमाल हुआ था। मुख्य file आम तौर पर यह होती है:

```bash
output/models-boogu-image-v0.1/pytorch_lora_weights.safetensors
```
