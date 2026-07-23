# Mage-Flow क्विकस्टार्ट

यह गाइड SimpleTuner में Mage-Flow LoRA ट्रेनिंग के लिए है। Mage-Flow Microsoft की 4B rectified-flow image generation और editing family है। यह native-resolution MMDiT transformer, Qwen3-VL text conditioning, और 128-channel, 16x downsample Mage-VAE का उपयोग करता है।

## हार्डवेयर

Mage-Flow Flux.1 और Qwen-Image से छोटा है, लेकिन फिर भी बड़ा transformer और frozen Qwen3-VL text encoder लोड करता है।

शुरुआती सेटिंग:

- `bf16`, 512px, batch 1 smoke test के लिए
- `bf16`, 1024px, batch 1 सामान्य LoRA के लिए
- VRAM कम हो तो `int8-torchao` या NF4
- Turbo flavours के लिए validation में 4 steps

24GB reduced या quantised प्रयोगों के लिए व्यावहारिक न्यूनतम है, 48GB 1024px के लिए बेहतर है, और 80GB edit training या बड़े batch के लिए अच्छा है।

## कॉन्फ़िगरेशन

SimpleTuner इंस्टॉल करें:

```bash
pip install 'simpletuner[cuda]'
```

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

Edit flavours को conditioning image dataset चाहिए। SimpleTuner `check_user_config` में Flux Kontext की तरह edit pipeline चुनता है।

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

Editing training के लिए source/target paired data का उपयोग करें। Caption edit instruction होना चाहिए, केवल target image description नहीं।

## Memory presets

Mage-Flow memory optimisation menu में RAMTorch और Musubi block swap presets देता है। Transformer weights को CPU RAM में रखना हो तो RAMTorch use करें; forward/backward के दौरान सिर्फ आखिरी Transformer blocks stream करने हों तो Musubi block swap use करें। Configurator में ये mutually exclusive हैं।

## Validation और quantisation

`default` के लिए लगभग 20 steps, `base` के लिए 30 steps, और `turbo` / `edit-turbo` के लिए 4 steps रखें।

```json
{
  "base_model_precision": "int8-torchao",
  "quantize_via": "cpu"
}
```

SimpleTuner MIT-licensed Mage-Flow code को vendor करता है और validation consistency के लिए native Diffusers pipelines में wrap करता है।
