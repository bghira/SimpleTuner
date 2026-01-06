# FLUX.2 क्विकस्टार्ट

यह गाइड FLUX.2-dev पर LoRAs ट्रेन करने को कवर करती है, जो Black Forest Labs का नवीनतम इमेज जनरेशन मॉडल है और इसमें Mistral-3 टेक्स्ट एन्कोडर शामिल है।

## मॉडल ओवरव्यू

FLUX.2-dev, FLUX.1 से बड़े आर्किटेक्चरल बदलाव लाता है:

- **Text Encoder**: CLIP+T5 की जगह Mistral-Small-3.1-24B
- **Architecture**: 8 DoubleStreamBlocks + 48 SingleStreamBlocks
- **Latent Channels**: 32 VAE चैनल → pixel shuffle के बाद 128 (FLUX.1 में 16)
- **VAE**: batch normalization और pixel shuffling वाला custom VAE
- **Embedding Dimension**: 15,360 (Mistral की layers 10, 20, 30 से stack)

## हार्डवेयर आवश्यकताएँ

Mistral-3 टेक्स्ट एन्कोडर के कारण FLUX.2 को काफी संसाधन चाहिए:

### VRAM आवश्यकताएँ

24B Mistral text encoder को अकेले भी काफी VRAM चाहिए:

| Component | bf16 | int8 | int4 |
|-----------|------|------|------|
| Mistral-3 (24B) | ~48GB | ~24GB | ~12GB |
| FLUX.2 Transformer | ~24GB | ~12GB | ~6GB |
| VAE + overhead | ~4GB | ~4GB | ~4GB |

| Configuration | Approximate Total VRAM |
|--------------|------------------------|
| bf16 everything | ~76GB+ |
| int8 text encoder + bf16 transformer | ~52GB |
| int8 everything | ~40GB |
| int4 text encoder + int8 transformer | ~22GB |

### System RAM

- **Minimum**: 96GB सिस्टम RAM (24B टेक्स्ट एन्कोडर लोड करने के लिए पर्याप्त मेमोरी चाहिए)
- **Recommended**: आरामदायक ऑपरेशन के लिए 128GB+

### अनुशंसित हार्डवेयर

- **Minimum**: 2x 48GB GPUs (A6000, L40S) FSDP2 या DeepSpeed के साथ
- **Recommended**: 4x H100 80GB with fp8-torchao
- **Heavy quantization (int4)** के साथ: 2x 24GB GPUs संभव हैं लेकिन experimental

Mistral-3 टेक्स्ट एन्कोडर और transformer के संयुक्त आकार के कारण FLUX.2 के लिए multi-GPU distributed training (FSDP2 या DeepSpeed) लगभग अनिवार्य है।

## प्रीरिक्विज़िट्स

### Python वर्ज़न

FLUX.2 को Python 3.10 या उससे नया और हाल के transformers चाहिए:

```bash
python --version  # Should be 3.10+
pip install transformers>=4.45.0
```

### मॉडल एक्सेस

FLUX.2-dev के लिए Hugging Face पर access approval चाहिए:

1. [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) पर जाएँ
2. License agreement स्वीकार करें
3. सुनिश्चित करें कि आप Hugging Face CLI में लॉगिन हैं

## इंस्टॉलेशन

```bash
pip install simpletuner[cuda]
```

Development setup के लिए:
```bash
git clone https://github.com/bghira/SimpleTuner
cd SimpleTuner
pip install -e ".[cuda]"
```

## कॉन्फ़िगरेशन

### Web Interface

```bash
simpletuner server
```

http://localhost:8001 खोलें और model family के रूप में FLUX.2 चुनें।

### Manual Configuration

`config/config.json` बनाएं:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "model_type": "lora",
  "model_family": "flux2",
  "model_flavour": "dev",
  "pretrained_model_name_or_path": "black-forest-labs/FLUX.2-dev",
  "output_dir": "/path/to/output",
  "train_batch_size": 1,
  "gradient_accumulation_steps": 1,
  "gradient_checkpointing": true,
  "mixed_precision": "bf16",
  "learning_rate": 1e-4,
  "lr_scheduler": "constant",
  "max_train_steps": 10000,
  "validation_resolution": "1024x1024",
  "validation_num_inference_steps": 20,
  "flux_guidance_mode": "constant",
  "flux_guidance_value": 1.0,
  "lora_rank": 16
}
```
</details>

### प्रमुख कॉन्फ़िगरेशन विकल्प

#### Guidance कॉन्फ़िगरेशन

FLUX.2, FLUX.1 की तरह guidance embedding उपयोग करता है:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "flux_guidance_mode": "constant",
  "flux_guidance_value": 1.0
}
```
</details>

या ट्रेनिंग के दौरान random guidance के लिए:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "flux_guidance_mode": "random-range",
  "flux_guidance_min": 1.0,
  "flux_guidance_max": 5.0
}
```
</details>

#### Quantization (Memory Optimization)

VRAM कम करने के लिए:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "int8-quanto",
  "base_model_default_dtype": "bf16"
}
```
</details>

#### TREAD (Training Acceleration)

FLUX.2 तेज़ ट्रेनिंग के लिए TREAD सपोर्ट करता है:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "tread_config": {
    "routes": [
      {"selection_ratio": 0.5, "start_layer_idx": 2, "end_layer_idx": -2}
    ]
  }
}
```
</details>

### उन्नत प्रयोगात्मक फीचर्स

<details>
<summary>उन्नत प्रयोगात्मक विवरण दिखाएँ</summary>


SimpleTuner में ऐसे प्रयोगात्मक फीचर्स शामिल हैं जो ट्रेनिंग स्थिरता और परफॉर्मेंस को काफी बेहतर कर सकते हैं।

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** exposure bias कम करता है और आउटपुट गुणवत्ता बेहतर करता है क्योंकि ट्रेनिंग के दौरान मॉडल अपने इनपुट्स स्वयं जनरेट करता है।

> ⚠️ ये फीचर्स ट्रेनिंग का कम्प्यूटेशनल ओवरहेड बढ़ाते हैं।

</details>

### डेटासेट कॉन्फ़िगरेशन

`config/multidatabackend.json` बनाएं:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
[
  {
    "id": "my-dataset",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/flux2/my-dataset",
    "instance_data_dir": "datasets/my-dataset",
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 10
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux2",
    "write_batch_size": 64
  }
]
```
</details>

> caption_strategy विकल्पों और आवश्यकताओं के लिए [DATALOADER.md](../DATALOADER.md#caption_strategy) देखें।

### वैकल्पिक edit / reference conditioning

FLUX.2 **plain text-to-image** (बिना conditioning) या **paired reference/edit images** के साथ ट्रेन कर सकता है। conditioning जोड़ने के लिए अपने मुख्य डेटासेट को एक या अधिक `conditioning` datasets से `conditioning_data` के जरिए जोड़ें और `conditioning_type` चुनें:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```jsonc
[
  {
    "id": "flux2-edits",
    "type": "local",
    "instance_data_dir": "/datasets/flux2/edits",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "conditioning_data": ["flux2-references"],
    "cache_dir_vae": "cache/vae/flux2/edits"
  },
  {
    "id": "flux2-references",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/flux2/references",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/flux2/references"
  }
]
```
</details>

- जब edit इमेज के साथ 1:1 crop alignment चाहिए, तब `conditioning_type=reference_strict` उपयोग करें। `reference_loose` mismatched aspect ratios की अनुमति देता है।
- edit और reference datasets के बीच फ़ाइल नामों का मेल होना चाहिए; हर edit इमेज की एक संबंधित reference फ़ाइल होनी चाहिए।
- कई conditioning datasets देने पर `conditioning_multidataset_sampling` (`combined` बनाम `random`) जरूरत अनुसार सेट करें; [OPTIONS](../OPTIONS.md#--conditioning_multidataset_sampling) देखें।
- `conditioning_data` के बिना, FLUX.2 standard text-to-image training पर लौट आता है।

### LoRA Targets

उपलब्ध LoRA target presets:

- `all` (डिफ़ॉल्ट): सभी attention और MLP layers
- `attention`: केवल attention layers (qkv, proj)
- `mlp`: केवल MLP/feed-forward layers
- `tiny`: न्यूनतम ट्रेनिंग (केवल qkv layers)

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "--flux_lora_target": "all"
}
```
</details>

## ट्रेनिंग

### सेवाओं में लॉगिन

```bash
huggingface-cli login
wandb login  # optional
```

### ट्रेनिंग शुरू करें

```bash
simpletuner train
```

या script से:

```bash
./train.sh
```

### Memory Offloading

Memory-constrained सेटअप्स के लिए, FLUX.2 transformer और वैकल्पिक रूप से Mistral-3 text encoder दोनों के लिए group offloading सपोर्ट करता है:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
--group_offload_text_encoder
```

`--group_offload_text_encoder` फ्लैग FLUX.2 के लिए अनुशंसित है क्योंकि 24B Mistral text encoder को text embedding caching के दौरान offloading से काफी लाभ मिलता है। आप `--group_offload_vae` भी जोड़ सकते हैं ताकि latent caching के दौरान VAE भी offload हो।

## Validation Prompts

`config/user_prompt_library.json` बनाएं:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "portrait_subject": "a professional portrait photograph of <subject>, studio lighting, high detail",
  "artistic_subject": "an artistic interpretation of <subject> in the style of renaissance painting",
  "cinematic_subject": "a cinematic shot of <subject>, dramatic lighting, film grain"
}
```
</details>

## Inference

### Trained LoRA का उपयोग

FLUX.2 LoRAs को SimpleTuner inference pipeline या community support विकसित होने पर compatible tools से लोड किया जा सकता है।

### Guidance Scale

- `flux_guidance_value=1.0` के साथ ट्रेनिंग अधिकांश use cases के लिए ठीक रहती है
- inference पर सामान्य guidance values (3.0-5.0) उपयोग करें

## FLUX.1 से अंतर

| Aspect | FLUX.1 | FLUX.2 |
|--------|--------|--------|
| Text Encoder | CLIP-L/14 + T5-XXL | Mistral-Small-3.1-24B |
| Embedding Dim | CLIP: 768, T5: 4096 | 15,360 (3×5,120) |
| Latent Channels | 16 | 32 (→128 after pixel shuffle) |
| VAE | AutoencoderKL | Custom (BatchNorm) |
| VAE Scale Factor | 8 | 16 (8×2 pixel shuffle) |
| Transformer Blocks | 19 joint + 38 single | 8 double + 48 single |

## Troubleshooting

### Startup के दौरान Out of Memory

- `--offload_during_startup=true` सक्षम करें
- text encoder quantization के लिए `--quantize_via=cpu` उपयोग करें
- `--vae_batch_size` घटाएँ

### धीमे text embeddings

Mistral-3 बड़ा है; विचार करें:
- ट्रेनिंग से पहले सभी text embeddings pre-cache करें
- text encoder quantization उपयोग करें
- बड़े `write_batch_size` के साथ batch processing

### ट्रेनिंग अस्थिरता

- learning rate कम करें (5e-5 आज़माएँ)
- gradient accumulation steps बढ़ाएँ
- gradient checkpointing सक्षम करें
- `--max_grad_norm=1.0` उपयोग करें

### CUDA Out of Memory

- quantization सक्षम करें (`int8-quanto` या `int4-quanto`)
- gradient checkpointing सक्षम करें
- batch size घटाएँ
- group offloading सक्षम करें
- token routing efficiency के लिए TREAD उपयोग करें

## Advanced: TREAD कॉन्फ़िगरेशन

TREAD (Token Routing for Efficient Architecture-agnostic Diffusion) चुनिंदा tokens प्रोसेस करके ट्रेनिंग को तेज़ करता है:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 4,
        "end_layer_idx": -4
      }
    ]
  }
}
```
</details>

- `selection_ratio`: रखें जाने वाले tokens का हिस्सा (0.5 = 50%)
- `start_layer_idx`: routing लागू करने की पहली layer
- `end_layer_idx`: आख़िरी layer (negative = end से)

Expected speedup: configuration पर निर्भर 20-40%.

## See Also

- [FLUX.1 Quickstart](FLUX.md) - FLUX.1 training के लिए
- [TREAD Documentation](../TREAD.md) - विस्तृत TREAD कॉन्फ़िगरेशन
- [LyCORIS Training Guide](../LYCORIS.md) - LoRA और LyCORIS प्रशिक्षण विधियाँ
- [Dataloader Configuration](../DATALOADER.md) - डेटासेट सेटअप
