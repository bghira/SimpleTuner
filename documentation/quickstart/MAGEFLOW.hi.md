# Mage-Flow क्विकस्टार्ट

यह गाइड SimpleTuner में Mage-Flow LoRA ट्रेनिंग के लिए है। Mage-Flow Microsoft की 4B rectified-flow image generation और editing family है। यह native-resolution MMDiT transformer, Qwen3-VL text conditioning, और 128-channel, 16x downsample Mage-VAE का उपयोग करता है।

## हार्डवेयर

Mage-Flow Flux.1 और Qwen-Image से छोटा है, लेकिन फिर भी बड़ा transformer और frozen Qwen3-VL text encoder लोड करता है।

शुरुआती सेटिंग:

- `bf16`, 512px, batch 1 smoke test के लिए
- `bf16`, 1024px, batch 1 सामान्य LoRA के लिए
- Ada/Hopper या नए NVIDIA GPU पर VRAM कम हो तो `fp8wo-torchao`
- Turbo flavours के लिए validation में 4 steps

24GB reduced या quantised प्रयोगों के लिए व्यावहारिक न्यूनतम है, 48GB 1024px के लिए बेहतर है, और 80GB edit training या बड़े batch के लिए अच्छा है।

## कॉन्फ़िगरेशन

SimpleTuner इंस्टॉल करें:

```bash
pip install 'simpletuner[cuda]'
```

Mage-Flow packed variable-length attention उपयोग करता है। Local `flash-attn` package build किए बिना FlashAttention 2 उपयोग करने के लिए `"attention_mechanism": "flash-attn-varlen-hub"` सेट करें, जिससे SimpleTuner Hugging Face Hub kernel लोड करेगा। PyTorch SDPA के लिए default `diffusers` value रहने दें।

Text-to-image शुरुआती config:

```json
{
  "model_family": "mageflow",
  "model_flavour": "base",
  "model_type": "lora",
  "pretrained_model_name_or_path": "microsoft/Mage-Flow-Base",
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "optimizer": "optimi-lion",
  "learning_rate": 1e-4,
  "lora_rank": 32,
  "train_batch_size": 1,
  "resolution": 1024,
  "validation_resolution": "1024x1024",
  "validation_num_inference_steps": 30,
  "validation_guidance": 5.0
}
```

Supported flavours:

- `base` - `microsoft/Mage-Flow-Base`
- `default` - `microsoft/Mage-Flow`
- `turbo` - `microsoft/Mage-Flow-Turbo`
- `edit-base` - `microsoft/Mage-Flow-Edit-Base`
- `edit` - `microsoft/Mage-Flow-Edit`
- `edit-turbo` - `microsoft/Mage-Flow-Edit-Turbo`

Editing के लिए:

```json
{
  "model_family": "mageflow",
  "model_flavour": "edit-turbo",
  "pretrained_model_name_or_path": "microsoft/Mage-Flow-Edit-Turbo",
  "validation_num_inference_steps": 4
}
```

## Mage Flow (Edit) Considerations

Mage-Flow edit checkpoints को conditioning या reference dataset की जरूरत नहीं है। Microsoft ने edit models को generation और editing tasks पर jointly train किया है, इसलिए generative prior सुरक्षित रहता है। SimpleTuner में `model_flavour` `edit-base`, `edit`, या `edit-turbo` होने पर भी आप subject, style, या concept LoRA finetuning के लिए सामान्य image dataset उपयोग कर सकते हैं।

Source/target paired data केवल तब उपयोग करें जब आप खास तौर पर edit behavior train करना चाहते हैं। SimpleTuner edit flavours के लिए edit-capable pipeline अपने आप उपयोग करता है; conditioning image न होने पर validation और prompt encoding text-to-image path इस्तेमाल करते हैं।

## Dataloader

Subject/style LoRA के लिए सामान्य image dataloader का उपयोग करें:

```json
[
  {
    "id": "dreambooth-1024",
    "type": "local",
    "instance_data_dir": "/path/to/images",
    "crop": true,
    "crop_style": "random",
    "crop_aspect": "square",
    "resolution": 1024,
    "resolution_type": "pixel",
    "metadata_backend": "discovery",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "cache_dir_vae": "cache/vae/mageflow/dreambooth-1024"
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/mageflow"
  }
]
```

Optional edit-behavior training के लिए source/target paired data का उपयोग करें। Caption edit instruction होना चाहिए, केवल target image description नहीं।

## Memory presets

Mage-Flow memory optimisation menu में RAMTorch और Musubi block swap presets देता है। Transformer weights को CPU RAM में रखना हो तो RAMTorch use करें; forward/backward के दौरान सिर्फ आखिरी Transformer blocks stream करने हों तो Musubi block swap use करें। Configurator में ये mutually exclusive हैं।

## Validation और quantisation

`default` के लिए लगभग 20 steps, `base` के लिए 30 steps, और `turbo` / `edit-turbo` के लिए 4 steps रखें।

```json
{
  "base_model_precision": "fp8wo-torchao",
  "quantize_via": "cpu"
}
```

SDNQ INT8 और ConvRot validate करते समय Mage-Flow के लिए पहले square-crop buckets prefer करें। Arbitrary aspect buckets में SDNQ को कई activation shapes compile/cache करने पड़ते हैं, जो short runs में dominant cost बन सकता है। H100 80GB Domokun LoRA smoke में center square crop ने सिर्फ एक `1.0` bucket इस्तेमाल किया और 100 steps पर stable loss ranges दिए:

| Precision path | Loss range | Mean loss | Loop s/step | Mean step |
| --- | ---: | ---: | ---: | ---: |
| BF16 | `0.117..1.17` | `0.5689` | `0.189` | `0.177` |
| FP8 weight-only TorchAO | `0.120..1.17` | `0.5696` | `0.234` | `0.222` |
| SDNQ INT8 vanilla | `0.129..1.17` | `0.5736` | `1.113` | `0.277` |
| SDNQ ConvRot 256 | `0.124..1.17` | `0.5696` | `0.436` | `0.299` |

उसी workload को arbitrary aspect buckets पर चलाने से SDNQ input shapes compile करने में काफी अधिक समय लगा। Text encoder frozen रखें; initial cache pass महंगा है, लेकिन normal LoRA runs में इसे train नहीं करना चाहिए। NF4 और दूसरे quantisation presets फिर भी उपयोगी हो सकते हैं।

SimpleTuner MIT-licensed Mage-Flow code को vendor करता है और validation consistency के लिए native Diffusers pipelines में wrap करता है।
