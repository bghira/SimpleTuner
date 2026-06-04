# Ideogram 4 क्विकस्टार्ट

यह गाइड SimpleTuner में Ideogram 4 LoRA ट्रेनिंग के लिए है। Ideogram 4 लगभग 9B पैरामीटर वाला flow-matching इमेज मॉडल है, जो typography, layout और complex prompts में मजबूत है। सार्वजनिक checkpoint FP8 weights के रूप में मिलता है; SimpleTuner इसे डिफ़ॉल्ट रूप से FP8 में उपयोग करता है।

शुरुआती config:

```bash
simpletuner/examples/ideogram-fp8.peft-lora/config.json
```

## हार्डवेयर और precision

अनुशंसित शुरुआत:

- **डिफ़ॉल्ट:** FP8 base weights, bf16 trainable LoRA weights, rank 16-32।
- **कम VRAM:** base model के लिए NF4।
- **अधिक VRAM:** पर्याप्त VRAM होने पर bf16-upcast weights, ताकि quantized loading से बचा जा सके।

80G NVIDIA GPU पर 1024px, batch size 1 के साथ FP8 या bf16-upcast LoRA training सामान्यतः फिट होनी चाहिए। छोटी GPUs पर FP8 या NF4, rank 8-16, gradient checkpointing और offload से शुरू करें। Apple GPUs Ideogram 4 training के लिए recommended नहीं हैं।

## कॉन्फ़िगरेशन

example config और dataloader कॉपी करें:

```bash
mkdir -p config/examples
cp simpletuner/examples/ideogram-fp8.peft-lora/config.json config/config.json
cp simpletuner/examples/multidatabackend-ideogram-dreambooth-1024px.json config/examples/multidatabackend-ideogram-dreambooth-1024px.json
```

मुख्य fields:

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
  "ideogram_schedule_std": 1.5
}
```

पहली recommendation FP8 है:

```json
{
  "model_flavour": "fp8",
  "base_model_precision": "no_change",
  "quantize_via": "cpu"
}
```

कम VRAM के लिए NF4:

```json
{
  "base_model_precision": "nf4-bnb",
  "base_model_default_dtype": "bf16",
  "quantize_via": "cpu"
}
```

## Validation

Ideogram validation opt-in है:

```json
{
  "ideogram_validation": true
}
```

यह temporary flag है। Ideogram का upstream CFG inference path अलग unconditional transformer expect करता है, जबकि SimpleTuner अभी default रूप से सिर्फ conditional transformer train करता है। इस flag के साथ validation conditional transformer को negative/unconditional pass के लिए भी use करता है, ताकि prompts और negative prompts जांचे जा सकें।

## Caption format

Ideogram 4 structured JSON captions के साथ बेहतर चलता है। Recommended fields:

- `high_level_description`
- `style_description`
- `style_description.color_palette`, hex colors के साथ
- `compositional_deconstruction.background`
- `compositional_deconstruction.elements`
- optional `bbox`, format `[x1, y1, x2, y2]`

अगर dataset में plain text और JSON captions दोनों हैं, इसे enabled रखें:

```json
{
  "ideogram_auto_json": true
}
```

Plain prompts Ideogram JSON schema में wrap होते हैं; existing JSON captions canonicalize होकर preserve रहती हैं। Hand-written JSON captions फिर भी बेहतर हैं, खासकर जब composition, background, elements और colors लिखे हों।

## Prompt upsampling

Optional:

```json
{
  "ideogram_prompt_upsample": true,
  "ideogram_prompt_enhancer_head_id": "diffusers/qwen3-vl-8b-instruct-lm-head"
}
```

यह JSON conversion से पहले Ideogram prompt upsampler से prompts rewrite करता है। यह slow है, इसलिए basic training path verify होने तक disabled रखें।

## LoRA और LyCORIS

Standard PEFT LoRA attention projections target करता है:

```json
{
  "lora_type": "standard",
  "lora_rank": 32
}
```

LyCORIS/LoKr Ideogram की `Attention` और `FeedForward` module classes target कर सकता है। Full-matrix LoKr बहुत बड़ा adapter बना सकता है; quick iteration के लिए standard LoRA से शुरू करें।

## Loss expectations

Ideogram loss दूसरे models की तुलना में high दिख सकता है। `1.0` के आसपास या उससे ऊपर values का मतलब यह नहीं कि model टूट गया है या validation images incoherent होंगी।

Tests में Ideogram ने coherent validation images बनाए, जबकि step loss लगभग `0.3-1.3` के बीच bounce करता रहा और occasional spikes भी आए। Run को validation image coherence, prompt adherence और loss exploding है या नहीं, इनसे judge करें।

## Training

```bash
simpletuner train
```

Development checkout:

```bash
CONFIG_BACKEND=json CONFIG_PATH=config/config.json .venv/bin/python simpletuner/train.py
```
