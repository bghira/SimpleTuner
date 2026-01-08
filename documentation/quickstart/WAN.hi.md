## Wan 2.1 क्विकस्टार्ट

इस उदाहरण में, हम Sayak Paul के [public domain Disney dataset](https://hf.co/datasets/sayakpaul/video-dataset-disney-organized) का उपयोग करके Wan 2.1 LoRA प्रशिक्षण करेंगे।



https://github.com/user-attachments/assets/51e6cbfd-5c46-407c-9398-5932fa5fa561


### हार्डवेयर आवश्यकताएँ

Wan 2.1 **1.3B** को बहुत अधिक सिस्टम **या** GPU मेमोरी की आवश्यकता नहीं होती। **14B** मॉडल भी समर्थित है, लेकिन वह काफी अधिक मांग वाला है।

वर्तमान में, Wan के लिए image‑to‑video प्रशिक्षण समर्थित नहीं है, लेकिन T2V LoRA और Lycoris I2V मॉडलों पर चलेंगे।

#### Text to Video

1.3B - https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
- Resolution: 832x480
- Rank‑16 LoRA लगभग 12G से थोड़ा अधिक उपयोग करता है (batch size 4)

14B - https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers
- Resolution: 832x480
- 24G में फिट हो जाएगा, लेकिन सेटिंग्स में थोड़ा बदलाव करना पड़ेगा।

<!--
#### Image to Video
14B (720p) - https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
- Resolution: 1280x720
-->

#### Image to Video (Wan 2.2)

हाल के Wan 2.2 I2V checkpoints उसी training flow के साथ काम करते हैं:

- High stage: https://huggingface.co/Wan-AI/Wan2.2-I2V-14B-Diffusers/tree/main/high_noise_model
- Low stage: https://huggingface.co/Wan-AI/Wan2.2-I2V-14B-Diffusers/tree/main/low_noise_model

आप इस गाइड में आगे बताए गए `model_flavour` और `wan_validation_load_other_stage` सेटिंग्स से इच्छित stage चुन सकते हैं।

आपको चाहिए:
- **यथार्थवादी न्यूनतम** 16GB या एक 3090 या V100 GPU
- **आदर्श रूप से** कई 4090, A6000, L40S, या बेहतर

यदि Wan 2.2 checkpoints चलाते समय time embedding layers में shape mismatches दिखें, तो नया
`wan_force_2_1_time_embedding` फ़्लैग सक्षम करें। यह transformer को Wan 2.1‑style time embeddings पर वापस ले जाता है और
compatibility समस्या हल करता है।

#### Stage presets और validation

- `model_flavour=i2v-14b-2.2-high` Wan 2.2 high‑noise stage को target करता है।
- `model_flavour=i2v-14b-2.2-low` low‑noise stage को target करता है (same checkpoints, अलग subfolder)।
- `wan_validation_load_other_stage=true` टॉगल करें ताकि validation renders के लिए training stage के विपरीत stage भी लोड हो।
- standard Wan 2.1 text‑to‑video रन के लिए flavour unset छोड़ें (या `t2v-480p-1.3b-2.1` उपयोग करें)।

Apple silicon सिस्टम्स अभी Wan 2.1 के साथ बहुत अच्छे नहीं चलते; एक training step में ~10 मिनट लग सकते हैं।

### पूर्वापेक्षाएँ

सुनिश्चित करें कि Python इंस्टॉल है; SimpleTuner 3.10 से 3.12 के साथ अच्छा काम करता है।

आप इसे चलाकर जांच सकते हैं:

```bash
python --version
```

यदि आपके Ubuntu पर Python 3.12 इंस्टॉल नहीं है, तो आप यह प्रयास कर सकते हैं:

```bash
apt -y install python3.12 python3.12-venv
```

#### Container image dependencies

Vast, RunPod, और TensorDock (आदि) के लिए, CUDA 12.2‑12.8 इमेज पर CUDA extensions कम्पाइल करने हेतु यह काम करेगा:

```bash
apt -y install nvidia-cuda-toolkit
```

### इंस्टॉलेशन

pip के जरिए SimpleTuner इंस्टॉल करें:

```bash
pip install simpletuner[cuda]
```

मैनुअल इंस्टॉलेशन या डेवलपमेंट सेटअप के लिए, [installation documentation](../INSTALL.md) देखें।

#### SageAttention 2

यदि आप SageAttention 2 उपयोग करना चाहते हैं, तो कुछ चरण करने होंगे।

> नोट: SageAttention न्यूनतम स्पीड‑अप देता है, बहुत प्रभावी नहीं; पता नहीं क्यों। 4090 पर टेस्ट किया गया है।

अपने Python venv के अंदर रहते हुए यह चलाएँ:
```bash
git clone https://github.com/thu-ml/SageAttention
pushd SageAttention
  pip install . --no-build-isolation
popd
```

#### AMD ROCm follow‑up steps

AMD MI300X को उपयोगी बनाने के लिए निम्न चलाना आवश्यक है:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

### वातावरण सेटअप

SimpleTuner चलाने के लिए, आपको एक configuration फ़ाइल, dataset और model directories, तथा एक dataloader configuration फ़ाइल सेट करनी होगी।

#### Configuration file

एक प्रयोगात्मक स्क्रिप्ट, `configure.py`, इंटरैक्टिव step‑by‑step कॉन्फ़िगरेशन के जरिए इस सेक्शन को पूरी तरह स्किप करने में मदद कर सकती है। इसमें कुछ सुरक्षा फीचर्स हैं जो सामान्य pitfalls से बचाते हैं।

**नोट:** यह आपके dataloader को कॉन्फ़िगर नहीं करता। आपको उसे बाद में मैन्युअली करना होगा।

इसे चलाने के लिए:

```bash
simpletuner configure
```

> ⚠️ जिन देशों में Hugging Face Hub आसानी से उपलब्ध नहीं है, वहाँ `HF_ENDPOINT=https://hf-mirror.com` को अपने `~/.bashrc` या `~/.zshrc` में जोड़ें, यह आपके सिस्टम के `$SHELL` पर निर्भर करता है।

### मेमोरी ऑफ़लोडिंग (वैकल्पिक)

Wan, SimpleTuner द्वारा समर्थित सबसे भारी मॉडलों में से एक है। यदि आप VRAM सीमा के पास हैं तो grouped offloading सक्षम करें:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- केवल CUDA डिवाइसेस `--group_offload_use_stream` को मानते हैं; ROCm/MPS स्वतः fallback करते हैं।
- disk staging तभी उपयोग करें जब CPU मेमोरी bottleneck हो।
- `--enable_model_cpu_offload` group offload के साथ mutually exclusive है।

### Feed‑forward chunking (वैकल्पिक)

यदि 14B checkpoints gradient checkpointing के बावजूद OOM करें, तो Wan feed‑forward layers को chunk करें:

```bash
--enable_chunked_feed_forward \
--feed_forward_chunk_size 2 \
```

यह कॉन्फ़िगरेशन विज़ार्ड (`Training → Memory Optimisation`) में नए टॉगल से मेल खाता है। छोटे chunk sizes अधिक
मेमोरी बचाते हैं लेकिन हर स्टेप धीमा करते हैं। आप त्वरित प्रयोगों के लिए `WAN_FEED_FORWARD_CHUNK_SIZE=2` भी सेट कर सकते हैं।


यदि आप मैन्युअल कॉन्फ़िगर करना पसंद करते हैं:

`config/config.json.example` को `config/config.json` में कॉपी करें:

```bash
cp config/config.json.example config/config.json
```

Multi‑GPU उपयोगकर्ता उपयोग किए जाने वाले GPU की संख्या कॉन्फ़िगर करने के लिए [इस दस्तावेज़](../OPTIONS.md#environment-configuration-variables) को देखें।

अंत में आपका config कुछ ऐसा दिखेगा:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "resume_from_checkpoint": "latest",
  "quantize_via": "cpu",
  "attention_mechanism": "sageattention",
  "data_backend_config": "config/wan/multidatabackend.json",
  "aspect_bucket_rounding": 2,
  "seed": 42,
  "minimum_image_size": 0,
  "offload_during_startup": true,
  "disable_benchmark": false,
  "output_dir": "output/wan",
  "lora_type": "standard",
  "lycoris_config": "config/wan/lycoris_config.json",
  "max_train_steps": 400000,
  "num_train_epochs": 0,
  "checkpoint_step_interval": 1000,
  "checkpoints_total_limit": 5,
  "hub_model_id": "wan-disney",
  "push_to_hub": "true",
  "push_checkpoints_to_hub": "true",
  "tracker_project_name": "lora-training",
  "tracker_run_name": "wan-adamW",
  "report_to": "wandb",
  "model_type": "lora",
  "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "pretrained_t5_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "model_family": "wan",
  "train_batch_size": 2,
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 1,
  "caption_dropout_probability": 0.1,
  "resolution_type": "pixel_area",
  "resolution": 480,
  "validation_seed": 42,
  "validation_step_interval": 100,
  "validation_resolution": "832x480",
  "validation_prompt": "两只拟人化的猫咪身穿舒适的拳击装备，戴着鲜艳的手套，在聚光灯照射的舞台上激烈对战",
  "validation_negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
  "validation_guidance": 5.2,
  "validation_num_inference_steps": 40,
  "validation_num_video_frames": 81,
  "mixed_precision": "bf16",
  "optimizer": "optimi-lion",
  "learning_rate": 0.00005,
  "max_grad_norm": 0.01,
  "grad_clip_method": "value",
  "lr_scheduler": "cosine",
  "lr_warmup_steps": 400000,
  "base_model_precision": "no_change",
  "vae_batch_size": 1,
  "webhook_config": "config/wan/webhook.json",
  "compress_disk_cache": true,
  "use_ema": true,
  "ema_validation": "ema_only",
  "ema_update_interval": 2,
  "delete_problematic_images": "true",
  "disable_bucket_pruning": true,
  "validation_guidance_skip_layers": [9],
  "validation_guidance_skip_layers_start": 0.0,
  "validation_guidance_skip_layers_stop": 1.0,
  "lora_rank": 16,
  "flow_schedule_shift": 3,
  "validation_prompt_library": false,
  "ignore_final_epochs": true
}
```
</details>

विशेष रूप से इस कॉन्फ़िग में validation settings महत्वपूर्ण हैं। इनके बिना आउटपुट अच्छे नहीं लगते।

### वैकल्पिक: CREPA temporal regularizer

Wan में स्मूद motion और कम identity drift के लिए:
- **Training → Loss functions** में **CREPA** सक्षम करें।
- **Block Index = 8**, **Weight = 0.5**, **Adjacent Distance = 1**, **Temporal Decay = 1.0** से शुरू करें।
- डिफ़ॉल्ट encoder (`dinov2_vitg14`, size `518`) अच्छा काम करता है; VRAM बचाने के लिए ही `dinov2_vits14` + `224` चुनें।
- पहली रन में torch hub से DINOv2 डाउनलोड होगा; offline प्रशिक्षण के लिए cache/prefetch करें।
- **Drop VAE Encoder** केवल तब सक्षम करें जब आप पूरी तरह cached latents से प्रशिक्षण कर रहे हों; अन्यथा इसे ऑफ रखें ताकि pixel encodes काम करें।

### उन्नत प्रयोगात्मक विशेषताएँ

<details>
<summary>उन्नत प्रयोगात्मक विवरण दिखाएँ</summary>


SimpleTuner में प्रयोगात्मक फीचर्स शामिल हैं जो प्रशिक्षण की स्थिरता और प्रदर्शन को काफी बेहतर कर सकते हैं।

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** exposure bias कम करता है और आउटपुट गुणवत्ता बढ़ाता है, क्योंकि यह प्रशिक्षण के दौरान मॉडल को अपने इनपुट्स खुद जनरेट करने देता है।

> ⚠️ ये फीचर्स प्रशिक्षण के कंप्यूटेशनल ओवरहेड को बढ़ाते हैं।

</details>

### TREAD प्रशिक्षण

> ⚠️ **Experimental**: TREAD एक नया फीचर है। यह कार्यात्मक है, लेकिन optimal कॉन्फ़िग्स अभी खोजे जा रहे हैं।

[TREAD](../TREAD.md) (paper) का अर्थ है **T**oken **R**outing for **E**fficient **A**rchitecture‑agnostic **D**iffusion। यह Flux प्रशिक्षण को तेज़ करने की विधि है जो transformer layers में tokens को बुद्धिमानी से route करती है। speedup इस बात के अनुपात में होता है कि आप कितने tokens drop करते हैं।

#### Quick setup

अपने `config.json` में यह जोड़ें ताकि bs=2 और 480p पर ~5 सेकंड प्रति step तक पहुँचने के लिए एक सरल और conservative तरीका मिले (vanilla ~10 सेकंड/step से कम):

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.1,
        "start_layer_idx": 2,
        "end_layer_idx": -2
      }
    ]
  }
}
```
</details>

यह कॉन्फ़िग:
- layers 2 से second‑to‑last तक केवल 50% image tokens रखता है
- text tokens कभी drop नहीं होते
- ~25% speedup, न्यूनतम गुणवत्ता प्रभाव के साथ
- training गुणवत्ता और convergence सुधारने की संभावना

Wan 1.3B के लिए, हम सभी 29 layers में progressive route setup से इस तरीके को बेहतर बना सकते हैं और bs=2, 480p पर ~7.7 सेकंड/step तक पहुँच सकते हैं:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "tread_config": {
      "routes": [
          { "selection_ratio": 0.1, "start_layer_idx": 2, "end_layer_idx": 8 },
          { "selection_ratio": 0.25, "start_layer_idx": 9, "end_layer_idx": 11 },
          { "selection_ratio": 0.35, "start_layer_idx": 12, "end_layer_idx": 15 },
          { "selection_ratio": 0.25, "start_layer_idx": 16, "end_layer_idx": 23 },
          { "selection_ratio": 0.1, "start_layer_idx": 24, "end_layer_idx": -2 }
      ]
  }
}
```
</details>

यह कॉन्फ़िग मॉडल की inner layers में अधिक aggressive token dropout लगाने की कोशिश करता है जहाँ semantic knowledge उतनी महत्वपूर्ण नहीं है।

कुछ डेटासेट्स पर अधिक aggressive dropout स्वीकार्य हो सकता है, लेकिन Wan 2.1 के लिए 0.5 का मान काफी ऊँचा है।

#### मुख्य बिंदु

- **सीमित आर्किटेक्चर सपोर्ट** - TREAD केवल Flux और Wan मॉडलों के लिए लागू है
- **उच्च resolutions पर सर्वोत्तम** - 1024x1024+ पर सबसे बड़े speedups, क्योंकि attention की O(n²) complexity होती है
- **Masked loss के साथ संगत** - masked regions स्वतः preserved रहते हैं (लेकिन speedup घटता है)
- **Quantization के साथ काम करता है** - int8/int4/NF4 training के साथ संयोजन संभव
- **शुरुआती loss spike की उम्मीद** - LoRA/LoKr प्रशिक्षण शुरू करते समय loss अधिक होगा लेकिन जल्दी सामान्य हो जाएगा

#### Tuning टिप्स

- **Conservative (quality‑focused)**: `selection_ratio` 0.1‑0.3 रखें
- **Aggressive (speed‑focused)**: `selection_ratio` 0.3‑0.5 रखें और गुणवत्ता प्रभाव स्वीकार करें
- **शुरुआती/अंतिम layers से बचें**: layers 0‑1 या अंतिम layer में routing न करें
- **LoRA training के लिए**: हल्की slowdowns हो सकती हैं — अलग‑अलग configs आज़माएँ
- **Resolution जितना ऊँचा, speedup उतना बेहतर**: 1024px और ऊपर पर सबसे लाभ

#### Known behavior

- जितने अधिक tokens drop (ऊँचा `selection_ratio`), training उतनी तेज़ लेकिन शुरुआती loss अधिक
- LoRA/LoKr training में शुरुआती loss spike दिखता है जो नेटवर्क adapt होते ही जल्दी ठीक हो जाता है
  - कम aggressive configuration या inner layers में अधिक levels वाले multiple routes उपयोग करने से यह कम होगा
- कुछ LoRA configs थोड़ा धीमा ट्रेन कर सकते हैं — optimal configs अभी खोजे जा रहे हैं
- RoPE (rotary position embedding) implementation functional है लेकिन 100% सही नहीं हो सकता

विस्तृत कॉन्फ़िग विकल्पों और troubleshooting के लिए [पूर्ण TREAD दस्तावेज़](../TREAD.md) देखें।


#### Validation prompts

`config/config.json` के अंदर "primary validation prompt" होता है, जो आमतौर पर आपके single subject या style के लिए मुख्य instance_prompt होता है। इसके अतिरिक्त, एक JSON फ़ाइल बनाई जा सकती है जिसमें वैलिडेशन के दौरान चलाने के लिए अतिरिक्त प्रॉम्प्ट्स हों।

उदाहरण config फ़ाइल `config/user_prompt_library.json.example` का फ़ॉर्मैट:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```
</details>

nicknames validation के लिए फ़ाइलनाम होते हैं, इसलिए इन्हें छोटा और फ़ाइलसिस्टम‑अनुकूल रखें।

ट्रेनर को इस prompt library की ओर इंगित करने के लिए, `config.json` के अंत में नया लाइन जोड़कर इसे TRAINER_EXTRA_ARGS में जोड़ें:
<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
  "--user_prompt_library": "config/user_prompt_library.json",
```
</details>

विविध प्रॉम्प्ट्स का सेट यह निर्धारित करने में मदद करेगा कि मॉडल प्रशिक्षण के दौरान collapse तो नहीं हो रहा। इस उदाहरण में `<token>` शब्द को अपने subject नाम (instance_prompt) से बदलें।

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
    "anime_<token>": "a breathtaking anime-style video featuring <token>, capturing her essence with vibrant colors, dynamic motion, and expressive storytelling",
    "chef_<token>": "a high-quality, detailed video of <token> as a sous-chef, immersed in the art of culinary creation with captivating close-ups and engaging sequences",
    "just_<token>": "a lifelike and intimate video portrait of <token>, showcasing her unique personality and charm through nuanced movement and expression",
    "cinematic_<token>": "a cinematic, visually stunning video of <token>, emphasizing her dramatic and captivating presence through fluid camera movements and atmospheric effects",
    "elegant_<token>": "an elegant and timeless video portrait of <token>, exuding grace and sophistication with smooth transitions and refined visuals",
    "adventurous_<token>": "a dynamic and adventurous video featuring <token>, captured in an exciting, action-filled sequence that highlights her energy and spirit",
    "mysterious_<token>": "a mysterious and enigmatic video portrait of <token>, shrouded in shadows and intrigue with a narrative that unfolds in subtle, cinematic layers",
    "vintage_<token>": "a vintage-style video of <token>, evoking the charm and nostalgia of a bygone era through sepia tones and period-inspired visual storytelling",
    "artistic_<token>": "an artistic and abstract video representation of <token>, blending creativity with visual storytelling through experimental techniques and fluid visuals",
    "futuristic_<token>": "a futuristic and cutting-edge video portrayal of <token>, set against a backdrop of advanced technology with sleek, high-tech visuals",
    "woman": "a beautifully crafted video portrait of a woman, highlighting her natural beauty and unique features through elegant motion and storytelling",
    "man": "a powerful and striking video portrait of a man, capturing his strength and character with dynamic sequences and compelling visuals",
    "boy": "a playful and spirited video portrait of a boy, capturing youthful energy and innocence through lively scenes and engaging motion",
    "girl": "a charming and vibrant video portrait of a girl, emphasizing her bright personality and joy with colorful visuals and fluid movement",
    "family": "a heartwarming and cohesive family video, showcasing the bonds and connections between loved ones through intimate moments and shared experiences"
}
```
</details>

> ℹ️ Wan 2.1 केवल UMT5 text encoder का उपयोग करता है, जिसमें embeddings में बहुत स्थानीय जानकारी होती है, इसलिए छोटे prompts में पर्याप्त जानकारी नहीं हो सकती। लंबे, वर्णनात्मक prompts उपयोग करें।

#### CLIP score ट्रैकिंग

वर्तमान समय में वीडियो मॉडल प्रशिक्षण के लिए यह सक्षम नहीं किया जाना चाहिए।

# स्थिर evaluation loss

यदि आप मॉडल प्रदर्शन स्कोर करने के लिए stable MSE loss उपयोग करना चाहते हैं, तो evaluation loss को कॉन्फ़िगर और इंटरप्रेट करने के लिए [यह दस्तावेज़](../evaluation/EVAL_LOSS.md) देखें।

#### Validation previews

SimpleTuner Tiny AutoEncoder मॉडलों का उपयोग करके generation के दौरान intermediate validation previews स्ट्रीम करने का समर्थन करता है। इससे आप webhook callbacks के जरिए real‑time में step‑by‑step validation images देख सकते हैं।

सक्रिय करने के लिए:
<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```
</details>

**आवश्यकताएँ:**
- Webhook configuration
- Validation सक्षम होना

`validation_preview_steps` को ऊँचा मान (जैसे 3 या 5) रखें ताकि Tiny AutoEncoder का ओवरहेड कम हो। `validation_num_inference_steps=20` और `validation_preview_steps=5` के साथ, आपको steps 5, 10, 15, और 20 पर preview images मिलेंगे।

#### Flow‑matching schedule shift

Flux, Sana, SD3, LTX Video और Wan 2.1 जैसे flow‑matching मॉडलों में `shift` नाम का गुण होता है जो हमें एक सरल decimal value से timestep schedule के प्रशिक्षित हिस्से को शिफ्ट करने देता है।

##### Defaults
डिफ़ॉल्ट रूप से कोई schedule shift नहीं लागू होता, जिससे timestep sampling distribution में sigmoid bell‑shape बनती है, जिसे `logit_norm` भी कहते हैं।

##### Auto‑shift
एक सामान्य रूप से अनुशंसित तरीका यह है कि resolution‑dependent timestep shift सक्षम किया जाए, `--flow_schedule_auto_shift`, जो बड़े images के लिए उच्च shift मान और छोटे images के लिए कम shift मान उपयोग करता है। इससे स्थिर लेकिन संभवतः औसत प्रशिक्षण परिणाम मिलते हैं।

##### Manual specification
_Discord के General Awareness का इन उदाहरणों के लिए धन्यवाद_

> ℹ️ ये उदाहरण Flux Dev के साथ value का प्रभाव दिखाते हैं, लेकिन Wan 2.1 भी काफ़ी समान होना चाहिए।

`--flow_schedule_shift` का मान 0.1 (बहुत कम) रखने पर केवल इमेज के सूक्ष्म विवरण प्रभावित होते हैं:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

`--flow_schedule_shift` का मान 4.0 (बहुत अधिक) रखने पर बड़े compositional features और संभवतः मॉडल का colour space प्रभावित हो सकता है:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)


#### Quantised model training

Apple और NVIDIA सिस्टम्स पर टेस्ट किया गया है; Hugging Face Optimum‑Quanto का उपयोग precision और VRAM आवश्यकताओं को घटाने के लिए किया जा सकता है, जिससे 16GB पर भी training संभव हो सके।



`config.json` उपयोगकर्ताओं के लिए:
<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "lora_rank": 16,
  "max_grad_norm": 1.0,
  "base_model_default_dtype": "bf16"
```
</details>

#### Validation settings

Wan 2.1 को SimpleTuner में जोड़ने के शुरुआती प्रयोगों के दौरान, आउटपुट बेहद खराब आ रहे थे, और इसके कुछ कारण हैं:

- inference के लिए पर्याप्त steps नहीं
  - यदि आप UniPC उपयोग नहीं कर रहे हैं, तो शायद कम से कम 40 steps चाहिए। UniPC थोड़ा कम कर सकता है, लेकिन प्रयोग करना होगा।
- गलत scheduler configuration
  - सामान्य Euler flow matching schedule उपयोग हो रहा था, लेकिन Betas distribution सबसे अच्छा काम करता है
  - यदि आपने यह सेटिंग नहीं बदली है, तो अब यह सही होनी चाहिए
- गलत resolution
  - Wan 2.1 असल में उन्हीं resolutions पर सही काम करता है जिन पर उसे ट्रेन किया गया था; कभी‑कभी भाग्य से काम करता है, लेकिन अक्सर खराब परिणाम आते हैं
- खराब CFG मान
  - Wan 2.1 1.3B खासकर CFG के प्रति संवेदनशील है, लेकिन 4.0‑5.0 के आसपास सुरक्षित लगता है
- खराब prompting
  - वीडियो मॉडल्स के लिए prompting सीखना मानो रहस्यमय कला है; उनके datasets और caption शैली को Holy Grail की तरह संभाला जाता है।
  - tl;dr अलग‑अलग prompts आज़माएँ।

इन सब के बावजूद, जब तक आपका batch size बहुत कम नहीं और/या learning rate बहुत ऊँचा नहीं, मॉडल आपके पसंदीदा inference टूल में सही चलेगा (मानते हुए कि आप पहले से अच्छे परिणाम पा रहे हैं)।

#### डेटासेट विचार

डेटासेट आकार पर सीमाएँ मुख्यतः compute और समय पर निर्भर हैं, जितना इसे प्रोसेस और ट्रेन करने में लगेगा।

सुनिश्चित करें कि डेटासेट मॉडल प्रशिक्षण के लिए पर्याप्त बड़ा हो, लेकिन आपकी उपलब्ध compute से बहुत बड़ा न हो।

ध्यान दें कि न्यूनतम डेटासेट आकार `train_batch_size * gradient_accumulation_steps` और `vae_batch_size` से अधिक होना चाहिए। यदि बहुत छोटा है, तो डेटासेट उपयोग योग्य नहीं होगा।

> ℹ️ बहुत कम samples होने पर आपको **no samples detected in dataset** संदेश दिख सकता है — `repeats` मान बढ़ाना इस सीमा को पार करेगा।

आपके डेटासेट के अनुसार, आपको डेटासेट डायरेक्टरी और dataloader configuration फ़ाइल अलग तरीके से सेट करनी होगी।

इस उदाहरण में हम [video-dataset-disney-organized](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized) डेटासेट उपयोग करेंगे।

एक `--data_backend_config` (`config/multidatabackend.json`) दस्तावेज़ बनाएँ जिसमें यह हो:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
[
  {
    "id": "disney-black-and-white",
    "type": "local",
    "dataset_type": "video",
    "crop": false,
    "resolution": 480,
    "minimum_image_size": 480,
    "maximum_image_size": 480,
    "target_downsample_size": 480,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/wan/disney-black-and-white",
    "instance_data_dir": "datasets/disney-black-and-white",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 75,
        "min_frames": 75,
        "bucket_strategy": "aspect_ratio"
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/wan",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> `caption_strategy` विकल्प और आवश्यकताओं के लिए [DATALOADER.md](../DATALOADER.md#caption_strategy) देखें।

- Wan 2.2 image‑to‑video runs CLIP conditioning caches बनाते हैं। **video** dataset entry में, एक dedicated backend की ओर इशारा करें और (वैकल्पिक रूप से) cache path override करें:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
  {
    "id": "disney-black-and-white",
    "type": "local",
    "dataset_type": "video",
    "conditioning_image_embeds": "disney-conditioning",
    "cache_dir_conditioning_image_embeds": "cache/conditioning_image_embeds/disney-black-and-white"
  }
```
</details>

- conditioning backend को एक बार परिभाषित करें और आवश्यकतानुसार datasets में reuse करें (यहाँ clarity के लिए full object दिखाया गया है):

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
  {
    "id": "disney-conditioning",
    "type": "local",
    "dataset_type": "conditioning_image_embeds",
    "cache_dir": "cache/conditioning_image_embeds/disney-conditioning",
    "disabled": false
  }
```
</details>

- `video` उप‑सेक्शन में हम ये keys सेट कर सकते हैं:
  - `num_frames` (वैकल्पिक, int) बताता है कि प्रशिक्षण में कितने फ्रेम उपयोग होंगे।
    - 15 fps पर 75 फ्रेम 5 सेकंड की वीडियो बनाते हैं, यह standard आउटपुट है। यही आपका target होना चाहिए।
  - `min_frames` (वैकल्पिक, int) प्रशिक्षण के लिए वीडियो की न्यूनतम लंबाई तय करता है।
    - यह कम से कम `num_frames` के बराबर होना चाहिए। इसे न सेट करने पर यह बराबर रखा जाता है।
  - `max_frames` (वैकल्पिक, int) प्रशिक्षण के लिए वीडियो की अधिकतम लंबाई तय करता है।
  - `bucket_strategy` (वैकल्पिक, string) वीडियो को buckets में समूहित करने का तरीका तय करता है:
    - `aspect_ratio` (डिफ़ॉल्ट): केवल spatial aspect ratio से समूहित (जैसे `1.78`, `0.75`).
    - `resolution_frames`: `WxH@F` फॉर्मैट (जैसे `832x480@75`) में resolution और फ्रेम काउंट के अनुसार समूहित। mixed‑resolution/duration datasets के लिए उपयोगी।
  - `frame_interval` (वैकल्पिक, int) `resolution_frames` उपयोग करते समय फ्रेम काउंट को इस इंटरवल तक राउंड करता है।
<!--  - `is_i2v` (वैकल्पिक, bool) बताता है कि डेटासेट पर i2v प्रशिक्षण किया जाएगा या नहीं।
    - यह Wan 2.1 के लिए डिफ़ॉल्ट रूप से True है। आप चाहें तो इसे बंद कर सकते हैं।
-->

फिर, `datasets` डायरेक्टरी बनाएँ:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset sayakpaul/video-dataset-disney-organized --local-dir=disney-black-and-white
popd
```

यह सभी Disney वीडियो सैंपल्स को आपकी `datasets/disney-black-and-white` डायरेक्टरी में डाउनलोड करेगा, जो अपने‑आप बन जाएगी।

#### WandB और Huggingface Hub में लॉग‑इन

प्रशिक्षण शुरू करने से पहले WandB और HF Hub में लॉग‑इन करना बेहतर है, खासकर यदि आप `--push_to_hub` और `--report_to=wandb` उपयोग कर रहे हैं।

यदि आप Git LFS रिपॉज़िटरी में मैन्युअली आइटम्स push करने वाले हैं, तो `git config --global credential.helper store` भी चलाएँ।

निम्न कमांड चलाएँ:

```bash
wandb login
```

और

```bash
huggingface-cli login
```

निर्देशों का पालन करके दोनों सेवाओं में लॉग‑इन करें।

### प्रशिक्षण रन निष्पादित करना

SimpleTuner डायरेक्टरी से, प्रशिक्षण शुरू करने के लिए आपके पास कई विकल्प हैं:

**विकल्प 1 (अनुशंसित - pip install):**
```bash
pip install simpletuner[cuda]
simpletuner train
```

**विकल्प 2 (Git clone विधि):**
```bash
simpletuner train
```

> ℹ️ जब आप Wan 2.2 ट्रेन करें, तो `TRAINER_EXTRA_ARGS` या CLI में `--model_flavour i2v-14b-2.2-high` (या `low`) जोड़ें और आवश्यकता हो तो `--wan_validation_load_other_stage` भी जोड़ें। `--wan_force_2_1_time_embedding` केवल तब जोड़ें जब checkpoint time‑embedding shape mismatch रिपोर्ट करे।

**विकल्प 3 (Legacy विधि - अभी भी काम करता है):**
```bash
./train.sh
```

इससे text embed और VAE आउटपुट कैशिंग डिस्क पर शुरू होगी।

अधिक जानकारी के लिए [dataloader](../DATALOADER.md) और [tutorial](../TUTORIAL.md) दस्तावेज़ देखें।

## नोट्स और समस्या‑समाधान टिप्स

### सबसे कम VRAM कॉन्फ़िग

Wan 2.1 quantisation के प्रति संवेदनशील है और फिलहाल NF4 या INT4 के साथ उपयोग नहीं किया जा सकता।

- OS: Ubuntu Linux 24
- GPU: एक NVIDIA CUDA डिवाइस (10G, 12G)
- System memory: लगभग 12G सिस्टम मेमोरी
- Base model precision: `int8-quanto`
- Optimiser: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolution: 480px
- Batch size: 1, zero gradient accumulation steps
- DeepSpeed: बंद/कॉन्फ़िगर नहीं
- PyTorch: 2.6
- `--gradient_checkpointing` ज़रूर सक्षम करें, वरना OOM रोकना संभव नहीं होगा
- केवल images पर ट्रेन करें, या अपने वीडियो डेटासेट के लिए `num_frames` को 1 सेट करें

**नोट**: VAE embeds और text encoder outputs की pre‑caching अधिक मेमोरी ले सकती है और फिर भी OOM हो सकता है। इसलिए `--offload_during_startup=true` लगभग अनिवार्य है। ऐसा होने पर text encoder quantisation और VAE tiling सक्षम किया जा सकता है। (Wan फिलहाल VAE tiling/slicing सपोर्ट नहीं करता)

गति:
- M3 Max Macbook Pro पर 665.8 sec/iter
- NVIDIA 4090 पर batch size 1 के साथ 2 sec/iter
- NVIDIA 4090 पर batch size 4 के साथ 11 sec/iter

### SageAttention

`--attention_mechanism=sageattention` उपयोग करने पर validation समय पर inference तेज़ हो सकता है।

**नोट**: यह अंतिम VAE decode चरण के साथ संगत नहीं है, और उस हिस्से को तेज़ नहीं करेगा।

### Masked loss

Wan 2.1 के साथ इसका उपयोग न करें।

### Quantisation
- 24G में इस मॉडल को ट्रेन करने के लिए quantisation आवश्यक नहीं है

### Image artifacts
Wan में Euler Betas flow‑matching schedule या (डिफ़ॉल्ट रूप से) UniPC multistep solver उपयोग करना आवश्यक है, जो उच्च‑order scheduler है और मजबूत predictions देता है।

अन्य DiT मॉडलों की तरह, यदि आप निम्न में से कुछ करते हैं, तो samples में square grid artifacts **आ** सकते हैं:
- कम गुणवत्ता डेटा पर overtrain करना
- बहुत ऊँचा learning rate उपयोग करना
- सामान्य overtraining (कम‑क्षमता नेटवर्क के साथ बहुत अधिक images)
- undertraining (उच्च‑क्षमता नेटवर्क के साथ बहुत कम images)
- अजीब aspect ratios या training data sizes का उपयोग

### Aspect bucketing
- Videos को images की तरह bucket किया जाता है।
- square crops पर बहुत देर तक training इस मॉडल को अधिक नुकसान नहीं करेगी। खुलकर प्रयोग करें, यह reliable है।
- दूसरी ओर, dataset के natural aspect buckets को उपयोग करने से inference समय पर वे shapes अधिक bias हो सकते हैं।
  - यह वांछनीय हो सकता है, क्योंकि यह cinematic जैसे aspect‑dependent styles को अन्य resolutions में bleed होने से रोकता है।
  - लेकिन यदि आप कई aspect buckets पर समान परिणाम सुधारना चाहते हैं, तो `crop_aspect=random` के साथ प्रयोग करना पड़ेगा, जिसमें अपने downsides हैं।
- इमेज डायरेक्टरी dataset को कई बार परिभाषित करके configs मिश्रित करने से बहुत अच्छे परिणाम और अच्छी generalization मिली है।

### Training custom fine‑tuned Wan 2.1 models

Hugging Face Hub पर कुछ fine‑tuned मॉडलों में पूरा directory structure नहीं होता, जिससे कुछ विकल्प सेट करने पड़ते हैं।

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
    "model_family": "wan",
    "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "pretrained_transformer_model_name_or_path": "path/to-the-other-model",
    "pretrained_vae_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "pretrained_transformer_subfolder": "none",
}
```
</details>

> Note: `pretrained_transformer_name_or_path` के लिए आप single‑file `.safetensors` पाथ भी दे सकते हैं
