# Krea2 क्विकस्टार्ट

यह गाइड SimpleTuner में Krea2 LoRA training के लिए है। Krea2 एक बड़ा flow-matching image transformer है जो Qwen-style text conditioning और Qwen Image VAE का उपयोग करता है। यह high-memory NVIDIA GPUs पर सबसे व्यावहारिक है।

शुरुआती example यहां है:

```bash
simpletuner/examples/krea2.peft-lora/config.json
```

## अनुशंसित शुरुआत

पहली run के लिए example config से शुरू करें और settings conservative रखें:

```json
{
  "model_family": "krea2",
  "model_flavour": "raw",
  "model_type": "lora",
  "pretrained_model_name_or_path": "krea/Krea-2-Raw",
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "fuse_qkv_projections": true,
  "train_batch_size": 1,
  "base_model_precision": "no_change"
}
```

Krea2 1024px-native image model है, लेकिन 512px और 768px fast iteration और dataset checks के लिए उपयोगी हैं। Run stable होने के बाद 1024px dataloader इस्तेमाल करें।

## Hardware Notes

हमारे परीक्षण में Krea2 80GB H100 पर bf16, 1024px, batch 1 में train हो सका। Compile off होने पर बड़े batches भी fit हुए, लेकिन compile graph/cudagraph memory बढ़ाता है और कई बड़े batch settings OOM हो जाते हैं।

TorchAO int8 weight-only VRAM को काफी कम करता है, लेकिन tested SimpleTuner training path में यह bf16 से तेज नहीं था। इसे तब उपयोग करें जब memory capacity step time से अधिक महत्वपूर्ण हो।

Recommendations:

- Model fit हो तो `bf16` उपयोग करें।
- Memory headroom चाहिए तो `int8-torchao` उपयोग करें।
- `gradient_checkpointing=true` रखें।
- `fuse_qkv_projections=true` रखें।
- `dynamo_backend=inductor`, `dynamo_mode=reduce-overhead`, और `dynamo_use_regional_compilation=true` केवल batch/resolution fit होने की पुष्टि के बाद enable करें।

## मुख्य Config Values

```json
{
  "model_family": "krea2",
  "model_flavour": "raw",
  "model_type": "lora",
  "pretrained_model_name_or_path": "krea/Krea-2-Raw",
  "base_model_precision": "no_change",
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "fuse_qkv_projections": true,
  "optimizer": "optimi-lion",
  "learning_rate": 1e-4,
  "lora_rank": 64,
  "train_batch_size": 1,
  "resolution": 1024,
  "validation_resolution": "1024x1024"
}
```

TorchAO int8 के लिए:

```json
{
  "base_model_precision": "int8-torchao",
  "quantize_via": "cpu"
}
```

Reduce-overhead compile के लिए:

```json
{
  "dynamo_backend": "inductor",
  "dynamo_mode": "reduce-overhead",
  "dynamo_use_regional_compilation": true
}
```

## Reference Image Training

Krea2 edit-style datasets के लिए optional reference-latent conditioning support करता है। जब dataloader paired reference images या cached reference latents देता हो, इसे enable करें:

```json
{
  "krea2_reference_latents": true
}
```

Reference latents का shape target latents से match होना चाहिए।

## Dataloader Configuration

Krea2 वही general image dataloader structure उपयोग करता है जो दूसरे image transformer models करते हैं। वास्तविक training resolution dataloader JSON से आती है, केवल main config के `resolution` से नहीं। 1024px training के लिए dataloader में `resolution`, `maximum_image_size`, और `target_downsample_size` भी 1024 होने चाहिए।

512px datasets fast tests, captions जांचने और crop समस्याएं पकड़ने के लिए उपयोगी हैं। Final quality signal के लिए 1024px run अधिक उपयोगी होता है।

Local datasets के लिए `type: local`, `instance_data_dir`, और caption strategy set करें। छोटे subject LoRA के लिए `caption_strategy=instanceprompt` अच्छी शुरुआत है। Style LoRA में filenames या full captions बेहतर हो सकते हैं।

## Validation

Krea2 validation महंगी है, इसलिए tuning के समय कम prompts रखें। एक prompt overfit या memorisation छिपा सकता है। Run stable होने के बाद छोटी prompt library जोड़ें।

```json
{
  "validation_prompt": "a studio portrait of <token>, soft directional light, detailed fabric texture",
  "validation_negative_prompt": "ugly, cropped, blurry, low-quality, mediocre average",
  "validation_num_inference_steps": 28,
  "validation_guidance": 4.5,
  "validation_resolution": "1024x1024"
}
```

## Quantisation Notes

`int8-torchao` transformer base weights को int8 में store करता है और ऊपर bf16 LoRA weights train करता है। H100 पर इससे VRAM काफी कम हुई, लेकिन tested training path में यह bf16 से धीमा था। इसे speed guarantee नहीं, capacity option समझें।

## Benchmark Results

ये measurements single NVIDIA H100 80GB पर लिए गए: real SimpleTuner trainer, Krea2 LoRA, fused QKV projections, gradient checkpointing, और छोटा Domokun dataset। VRAM को `nvidia-smi` से externally sample किया गया। इन्हें केवल comparative guidance मानें; PyTorch, CUDA, driver, dataset, LoRA rank, optimizer, attention backend और GPU बदलने से values बदल सकती हैं।

### Fused QKV + Checkpointing, Compile Off

| Precision | Resolution | Batch | Steady s/step | Peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| bf16 | 512 | 1 | 0.353 | 31.10 GiB |
| bf16 | 512 | 4 | 1.230 | 39.31 GiB |
| bf16 | 512 | 8 | 2.430 | 50.32 GiB |
| bf16 | 1024 | 1 | 0.990 | 33.28 GiB |
| bf16 | 1024 | 4 | 3.850 | 48.35 GiB |
| bf16 | 1024 | 8 | 7.690 | 67.88 GiB |
| int8-torchao | 512 | 1 | 0.535 | 18.10 GiB |
| int8-torchao | 512 | 4 | 1.690 | 27.46 GiB |
| int8-torchao | 512 | 8 | 3.220 | 40.52 GiB |
| int8-torchao | 1024 | 1 | 1.330 | 20.35 GiB |
| int8-torchao | 1024 | 4 | 4.850 | 36.99 GiB |
| int8-torchao | 1024 | 8 | 9.520 | 58.84 GiB |

### Fused QKV + Checkpointing + Reduce-Overhead Compile

| Precision | Resolution | Batch | Status | Steady s/step | Peak VRAM |
| --- | ---: | ---: | --- | ---: | ---: |
| bf16 | 512 | 1 | ok | 0.260 | 41.20 GiB |
| bf16 | 512 | 4 | OOM | - | 79.07 GiB |
| bf16 | 512 | 8 | OOM | - | 79.10 GiB |
| bf16 | 1024 | 1 | ok | 0.704 | 63.71 GiB |
| bf16 | 1024 | 4 | OOM | - | 79.11 GiB |
| bf16 | 1024 | 8 | OOM | - | 78.40 GiB |
| int8-torchao | 512 | 1 | ok | 0.410 | 30.93 GiB |
| int8-torchao | 512 | 4 | ok | 1.300 | 78.60 GiB |
| int8-torchao | 512 | 8 | OOM | - | 79.12 GiB |
| int8-torchao | 1024 | 1 | ok | 0.990 | 58.68 GiB |
| int8-torchao | 1024 | 4 | OOM | - | 78.92 GiB |
| int8-torchao | 1024 | 8 | OOM | - | 78.09 GiB |

## Practical Guidance

- H100 single-GPU fast iteration के लिए bf16, fused QKV, checkpointing, compile on, batch 1 उपयोग करें।
- बड़े effective batches के लिए uncompiled bf16 बेहतर है; `train_batch_size` को VRAM limit तक बढ़ाएं।
- Memory-constrained runs के लिए `int8-torchao` उपयोग करें; VRAM कम होगी, लेकिन steps धीमे हो सकते हैं।
- Compile batch 1 में उपयोगी है, लेकिन VRAM बहुत बढ़ाकर बड़े batches fail कर सकता है।

## Common Issues

- यदि आपने 1024px अपेक्षित किया लेकिन log 512px दिखाता है, dataloader JSON जांचें।
- यदि compile OOM करता है लेकिन uncompiled run fit होता है, batch size घटाएं या compile बंद करें।
- यदि int8 कम VRAM इस्तेमाल करता है लेकिन धीमा है, यह हमारे H100 tests से मेल खाता है।
- यदि reference image validation को प्रभावित नहीं कर रही है, `krea2_reference_latents=true` और paired validation dataset जांचें।
- यदि model जल्दी overfit करता है, learning rate घटाएं, steps घटाएं, या dataset variety बढ़ाएं।
