## Sana Video क्विकस्टार्ट

इस उदाहरण में, हम Sana Video 2B 480p मॉडल प्रशिक्षण करेंगे।

### हार्डवेयर आवश्यकताएँ

Sana Video Wan autoencoder का उपयोग करता है और डिफ़ॉल्ट रूप से 480p पर 81‑frame sequences प्रोसेस करता है। मेमोरी उपयोग अन्य वीडियो मॉडलों के समान अपेक्षित है; gradient checkpointing पहले ही सक्षम करें और VRAM headroom सत्यापित करने के बाद ही `train_batch_size` बढ़ाएँ।

### मेमोरी ऑफ़लोडिंग (वैकल्पिक)

यदि आप VRAM सीमा के करीब हैं, तो config में grouped offloading सक्षम करें:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- CUDA उपयोगकर्ताओं को `--group_offload_use_stream` का लाभ मिलता है; अन्य backends इसे स्वतः अनदेखा करते हैं।
- `--group_offload_to_disk_path` तभी उपयोग करें जब सिस्टम RAM सीमित हो — disk staging धीमा है लेकिन रन स्थिर रखता है।
- group offloading उपयोग करते समय `--enable_model_cpu_offload` बंद रखें।

### पूर्वापेक्षाएँ

सुनिश्चित करें कि Python इंस्टॉल है; SimpleTuner 3.10 से 3.12 के साथ अच्छा काम करता है।

आप इसे चलाकर जांच सकते हैं:

```bash
python --version
```

यदि आपके Ubuntu पर Python 3.12 इंस्टॉल नहीं है, तो आप यह प्रयास कर सकते हैं:

```bash
apt -y install python3.13 python3.13-venv
```

#### Container image dependencies

Vast, RunPod, और TensorDock (आदि) के लिए, CUDA 12.2‑12.8 इमेज पर CUDA extensions कम्पाइल करने हेतु यह काम करेगा:

```bash
apt -y install nvidia-cuda-toolkit
```

### इंस्टॉलेशन

pip के जरिए SimpleTuner इंस्टॉल करें:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
```

मैनुअल इंस्टॉलेशन या डेवलपमेंट सेटअप के लिए, [installation documentation](../INSTALL.md) देखें।

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

यदि आप मैन्युअल कॉन्फ़िगर करना पसंद करते हैं:

`config/config.json.example` को `config/config.json` में कॉपी करें:

```bash
cp config/config.json.example config/config.json
```

फिर, आपको निम्न वेरिएबल्स बदलने होंगे:

- `model_type` - इसे `full` पर सेट करें।
- `model_family` - इसे `sanavideo` पर सेट करें।
- `pretrained_model_name_or_path` - इसे `Efficient-Large-Model/SANA-Video_2B_480p_diffusers` पर सेट करें।
- `pretrained_vae_model_name_or_path` - इसे `Efficient-Large-Model/SANA-Video_2B_480p_diffusers` पर सेट करें।
- `output_dir` - इसे उस डायरेक्टरी पर सेट करें जहाँ आप अपने checkpoints और validation videos रखना चाहते हैं। यहाँ full path उपयोग करने की सलाह है।
- `train_batch_size` - वीडियो प्रशिक्षण के लिए कम से शुरू करें और VRAM उपयोग सत्यापित करने के बाद ही बढ़ाएँ।
- `validation_resolution` - Sana Video 480p मॉडल है; `832x480` या जिन aspect buckets पर validation करना है, वे उपयोग करें।
- `validation_num_video_frames` - डिफ़ॉल्ट sampler length से मेल करने के लिए `81` सेट करें।
- `validation_guidance` - inference में Sana Video के लिए जिस मान के साथ आप सहज हों, वही रखें।
- `validation_num_inference_steps` - स्थिर गुणवत्ता के लिए लगभग 50 रखें।
- `framerate` - यदि छोड़ा जाए, तो Sana Video 16 fps डिफ़ॉल्ट लेता है; इसे अपने dataset से मेल कराएँ।

- `optimizer` - आप कोई भी optimiser उपयोग कर सकते हैं जिसे आप जानते हों, लेकिन इस उदाहरण में हम `optimi-adamw` इस्तेमाल करेंगे।
- `mixed_precision` - सबसे कुशल training के लिए `bf16` अनुशंसित है, या `no` (लेकिन मेमोरी अधिक खपत होगी और धीमा रहेगा)।
- `gradient_checkpointing` - VRAM उपयोग नियंत्रित करने के लिए इसे सक्षम करें।
- `use_ema` - इसे `true` सेट करने से मुख्य trained checkpoint के साथ अधिक स्मूद परिणाम मिलते हैं।

Multi‑GPU उपयोगकर्ता उपयोग किए जाने वाले GPU की संख्या कॉन्फ़िगर करने के लिए [इस दस्तावेज़](../OPTIONS.md#environment-configuration-variables) को देखें।

अंत में, आपका कॉन्फ़िग कुछ ऐसा दिखेगा:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "resume_from_checkpoint": "latest",
  "quantize_via": "cpu",
  "data_backend_config": "config/sanavideo/multidatabackend.json",
  "seed": 42,
  "output_dir": "output/sanavideo",
  "max_train_steps": 400000,
  "checkpoint_step_interval": 1000,
  "checkpoints_total_limit": 5,
  "tracker_project_name": "video-training",
  "tracker_run_name": "sanavideo-2b-480p",
  "report_to": "wandb",
  "model_type": "full",
  "pretrained_model_name_or_path": "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
  "pretrained_vae_model_name_or_path": "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
  "model_family": "sanavideo",
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 1,
  "caption_dropout_probability": 0.1,
  "resolution_type": "pixel_area",
  "resolution": 480,
  "validation_seed": 42,
  "validation_step_interval": 200,
  "validation_resolution": "832x480",
  "validation_negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
  "validation_guidance": 6.0,
  "validation_num_inference_steps": 50,
  "validation_num_video_frames": 81,
  "validation_prompt": "A short video of a small, fluffy animal exploring a sunny room with soft window light and gentle camera motion.",
  "mixed_precision": "bf16",
  "optimizer": "adamw_bf16",
  "learning_rate": 0.00005,
  "max_grad_norm": 0.0,
  "grad_clip_method": "value",
  "lr_scheduler": "cosine",
  "lr_warmup_steps": 400000,
  "base_model_precision": "bf16",
  "vae_batch_size": 1,
  "compress_disk_cache": true,
  "use_ema": true,
  "ema_validation": "ema_only",
  "ema_update_interval": 2,
  "delete_problematic_images": "true",
  "framerate": 16,
  "validation_prompt_library": false,
  "ignore_final_epochs": true
}
```
</details>

### वैकल्पिक: CREPA temporal regularizer

यदि आपकी वीडियो में flicker या drifting subjects दिखें, तो CREPA सक्षम करें:
- **Training → Loss functions** में **CREPA** ऑन करें।
- सुझाए गए डिफ़ॉल्ट्स: **Block Index = 10**, **Weight = 0.5**, **Adjacent Distance = 1**, **Temporal Decay = 1.0**.
- डिफ़ॉल्ट encoder (`dinov2_vitg14`, size `518`) रखें, जब तक आपको छोटा विकल्प (`dinov2_vits14` + `224`) VRAM बचाने के लिए न चाहिए।
- पहली रन में torch hub से DINOv2 डाउनलोड होगा; offline हों तो cache/prefetch करें।
- **Drop VAE Encoder** केवल तब ऑन करें जब आप पूरी तरह cached latents से training कर रहे हों; यदि आप अभी भी pixels encode कर रहे हैं तो इसे ऑफ रखें।

### उन्नत प्रयोगात्मक विशेषताएँ

<details>
<summary>उन्नत प्रयोगात्मक विवरण दिखाएँ</summary>


SimpleTuner में प्रयोगात्मक फीचर्स शामिल हैं जो प्रशिक्षण की स्थिरता और प्रदर्शन को काफी बेहतर कर सकते हैं।

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** exposure bias कम करता है और आउटपुट गुणवत्ता बढ़ाता है, क्योंकि यह प्रशिक्षण के दौरान मॉडल को अपने इनपुट्स खुद जनरेट करने देता है।

> ⚠️ ये फीचर्स प्रशिक्षण के कंप्यूटेशनल ओवरहेड को बढ़ाते हैं।

#### वैलिडेशन प्रॉम्प्ट्स

`config/config.json` के अंदर "primary validation prompt" होता है, जो आमतौर पर आपके single subject या style के लिए मुख्य instance_prompt होता है। इसके अतिरिक्त, एक JSON फ़ाइल बनाई जा सकती है जिसमें वैलिडेशन के दौरान चलाने के लिए अतिरिक्त प्रॉम्प्ट्स हों।

उदाहरण config फ़ाइल `config/user_prompt_library.json.example` का फ़ॉर्मैट:

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

nicknames validation के लिए फ़ाइलनाम होते हैं, इसलिए इन्हें छोटा और फ़ाइलसिस्टम‑अनुकूल रखें।

ट्रेनर को इस prompt library की ओर इंगित करने के लिए, `config.json` के अंत में नया लाइन जोड़कर इसे TRAINER_EXTRA_ARGS में जोड़ें:

```json
  "--user_prompt_library": "config/user_prompt_library.json",
```

विविध प्रॉम्प्ट्स का सेट यह निर्धारित करने में मदद करेगा कि मॉडल प्रशिक्षण के दौरान collapse तो नहीं हो रहा। इस उदाहरण में `<token>` शब्द को अपने subject नाम (instance_prompt) से बदलें।

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

> ℹ️ Sana Video एक flow‑matching मॉडल है; छोटे prompts में पर्याप्त जानकारी नहीं हो सकती। संभव हो तो वर्णनात्मक prompts उपयोग करें।

#### CLIP score ट्रैकिंग

वर्तमान समय में वीडियो मॉडल प्रशिक्षण के लिए यह सक्षम नहीं किया जाना चाहिए।

</details>

# स्थिर evaluation loss

यदि आप मॉडल प्रदर्शन स्कोर करने के लिए stable MSE loss उपयोग करना चाहते हैं, तो evaluation loss को कॉन्फ़िगर और इंटरप्रेट करने के लिए [यह दस्तावेज़](../evaluation/EVAL_LOSS.md) देखें।

#### Validation previews

SimpleTuner Tiny AutoEncoder मॉडलों का उपयोग करके generation के दौरान intermediate validation previews स्ट्रीम करने का समर्थन करता है। इससे आप webhook callbacks के जरिए real‑time में step‑by‑step validation videos देख सकते हैं।

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

`validation_preview_steps` को ऊँचा मान (जैसे 3 या 5) रखें ताकि Tiny AutoEncoder का ओवरहेड कम हो। `validation_num_inference_steps=20` और `validation_preview_steps=5` के साथ, आपको steps 5, 10, 15, और 20 पर preview frames मिलेंगे।

#### Flow‑matching schedule

Sana Video checkpoint से canonical flow‑matching schedule उपयोग करता है। user‑provided shift overrides को अनदेखा किया जाता है; इस मॉडल के लिए `flow_schedule_shift` और `flow_schedule_auto_shift` unset रखें।

#### Quantised model training

Precision विकल्प (bf16, int8, fp8) config में उपलब्ध हैं; इन्हें अपने हार्डवेयर के अनुसार मिलाएँ और instability आने पर उच्च precision पर लौटें।

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
    "cache_dir_vae": "cache/vae/sanavideo/disney-black-and-white",
    "instance_data_dir": "datasets/disney-black-and-white",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 81,
        "min_frames": 81,
        "bucket_strategy": "aspect_ratio"
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/sanavideo",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> `caption_strategy` विकल्प और आवश्यकताओं के लिए [DATALOADER.md](../DATALOADER.md#caption_strategy) देखें।

- `video` उप‑सेक्शन में हम ये keys सेट कर सकते हैं:
  - `num_frames` (वैकल्पिक, int) बताता है कि प्रशिक्षण में कितने फ्रेम उपयोग होंगे।
  - `min_frames` (वैकल्पिक, int) प्रशिक्षण के लिए वीडियो की न्यूनतम लंबाई तय करता है।
  - `max_frames` (वैकल्पिक, int) प्रशिक्षण के लिए वीडियो की अधिकतम लंबाई तय करता है।
  - `is_i2v` (वैकल्पिक, bool) बताता है कि डेटासेट पर i2v प्रशिक्षण किया जाएगा या नहीं।
  - `bucket_strategy` (वैकल्पिक, string) वीडियो को buckets में समूहित करने का तरीका तय करता है:
    - `aspect_ratio` (डिफ़ॉल्ट): केवल spatial aspect ratio से समूहित (जैसे `1.78`, `0.75`).
    - `resolution_frames`: `WxH@F` फॉर्मैट (जैसे `832x480@81`) में resolution और फ्रेम काउंट के अनुसार समूहित। mixed‑resolution/duration datasets के लिए उपयोगी।
  - `frame_interval` (वैकल्पिक, int) `resolution_frames` उपयोग करते समय फ्रेम काउंट को इस इंटरवल तक राउंड करता है।

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
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
simpletuner train
```

**विकल्प 2 (Git clone विधि):**

```bash
simpletuner train
```

**विकल्प 3 (Legacy विधि - अभी भी काम करता है):**

```bash
./train.sh
```

इससे text embed और VAE आउटपुट कैशिंग डिस्क पर शुरू होगी।

अधिक जानकारी के लिए [dataloader](../DATALOADER.md) और [tutorial](../TUTORIAL.md) दस्तावेज़ देखें।

## नोट्स और समस्या‑समाधान टिप्स

### Validation defaults

- यदि validation settings नहीं दी जातीं, तो Sana Video डिफ़ॉल्ट रूप से 81 frames और 16 fps उपयोग करता है।
- Wan autoencoder का पाथ base मॉडल पाथ से मेल खाना चाहिए; load‑time errors से बचने के लिए इन्हें aligned रखें।

### Masked loss

यदि आप किसी subject या style को ट्रेन कर रहे हैं और इनमें से किसी को mask करना चाहते हैं, तो Dreambooth गाइड के [masked loss training](../DREAMBOOTH.md#masked-loss) सेक्शन देखें।
