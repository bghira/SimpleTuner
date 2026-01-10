## LTX Video क्विकस्टार्ट

इस उदाहरण में, हम Sayak Paul के [public domain Disney dataset](https://hf.co/datasets/sayakpaul/video-dataset-disney-organized) का उपयोग करके LTX‑Video LoRA प्रशिक्षण करेंगे।

### हार्डवेयर आवश्यकताएँ

LTX को बहुत अधिक सिस्टम **या** GPU मेमोरी की आवश्यकता नहीं होती।

जब आप rank‑16 LoRA के हर घटक (MLP, projections, multimodal blocks) को ट्रेन करते हैं, तो यह M3 Mac पर लगभग 12G से थोड़ा अधिक उपयोग करता है (batch size 4)।

आपको चाहिए:
- **यथार्थवादी न्यूनतम** 16GB या एक 3090 या V100 GPU
- **आदर्श रूप से** कई 4090, A6000, L40S, या बेहतर

Apple silicon सिस्टम्स अब तक LTX के साथ काफी अच्छे चलते हैं, हालांकि Pytorch के MPS backend की सीमाओं के कारण कम resolution पर।

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
- `--group_offload_to_disk_path` तभी उपयोग करें जब सिस्टम RAM <64 GB हो — disk staging धीमा है लेकिन runs स्थिर रखता है।
- group offloading उपयोग करते समय `--enable_model_cpu_offload` बंद रखें।

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
pip install 'simpletuner[cuda]'
```

मैनुअल इंस्टॉलेशन या डेवलपमेंट सेटअप के लिए, [installation documentation](../INSTALL.md) देखें।

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

यदि आप मैन्युअल कॉन्फ़िगर करना पसंद करते हैं:

`config/config.json.example` को `config/config.json` में कॉपी करें:

```bash
cp config/config.json.example config/config.json
```

फिर, आपको संभवतः निम्न वेरिएबल्स बदलने होंगे:

- `model_type` - इसे `lora` पर सेट करें।
- `model_family` - इसे `ltxvideo` पर सेट करें।
- `pretrained_model_name_or_path` - इसे `Lightricks/LTX-Video-0.9.5` पर सेट करें।
- `pretrained_vae_model_name_or_path` - इसे `Lightricks/LTX-Video-0.9.5` पर सेट करें।
- `output_dir` - इसे उस डायरेक्टरी पर सेट करें जहाँ आप अपने checkpoints और validation images रखना चाहते हैं। यहाँ full path उपयोग करने की सलाह है।
- `train_batch_size` - स्थिरता के लिए बढ़ाया जा सकता है, लेकिन शुरुआत में 4 ठीक है
- `validation_resolution` - इसे LTX के सामान्य उपयोग वाले resolution पर सेट करें (`768x512`)
  - कई resolutions को कॉमा से अलग कर सकते हैं: `1280x768,768x512`
- `validation_guidance` - LTX inference में जिस मान के साथ आप सहज हों, वही रखें।
- `validation_num_inference_steps` - समय बचाने के लिए लगभग 25 रखें, फिर भी ठीक गुणवत्ता मिलती है।
- यदि आप LoRA का आकार काफी कम करना चाहते हैं, तो `--lora_rank=4` उपयोग करें। इससे VRAM कम होगा लेकिन सीखने की क्षमता घटेगी।

- `gradient_accumulation_steps` - update steps कई चरणों में जमा करेगा।
  - इससे training runtime linearly बढ़ेगा; 2 का मतलब आधी गति और दोगुना समय।
- `optimizer` - शुरुआती उपयोगकर्ताओं को adamw_bf16 अनुशंसित है, हालांकि optimi-lion और optimi-stableadamw भी अच्छे हैं।
- `mixed_precision` - शुरुआती उपयोगकर्ताओं को `bf16` पर रहना चाहिए
- `gradient_checkpointing` - लगभग हर स्थिति में true रखें
- `gradient_checkpointing_interval` - यह LTX Video पर अभी समर्थित नहीं है, इसलिए config से हटाएँ।

Multi‑GPU उपयोगकर्ता उपयोग किए जाने वाले GPU की संख्या कॉन्फ़िगर करने के लिए [इस दस्तावेज़](../OPTIONS.md#environment-configuration-variables) को देखें।

अंत में, आपका config कुछ ऐसा दिखेगा:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "resume_from_checkpoint": "latest",
  "quantize_via": "cpu",
  "data_backend_config": "config/ltxvideo/multidatabackend.json",
  "aspect_bucket_rounding": 2,
  "seed": 42,
  "minimum_image_size": 0,
  "disable_benchmark": false,
  "offload_during_startup": true,
  "output_dir": "output/ltxvideo",
  "lora_type": "lycoris",
  "lycoris_config": "config/ltxvideo/lycoris_config.json",
  "max_train_steps": 400000,
  "num_train_epochs": 0,
  "checkpoint_step_interval": 1000,
  "checkpoints_total_limit": 5,
  "hub_model_id": "ltxvideo-disney",
  "push_to_hub": "true",
  "push_checkpoints_to_hub": "true",
  "tracker_project_name": "lora-training",
  "tracker_run_name": "ltxvideo-adamW",
  "report_to": "wandb",
  "model_type": "lora",
  "pretrained_model_name_or_path": "Lightricks/LTX-Video-0.9.5",
  "model_family": "ltxvideo",
  "train_batch_size": 8,
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 1,
  "caption_dropout_probability": 0.1,
  "resolution_type": "pixel_area",
  "resolution": 800,
  "validation_seed": 42,
  "validation_step_interval": 100,
  "validation_resolution": "768x512",
  "validation_negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
  "validation_guidance": 3.0,
  "validation_num_inference_steps": 40,
  "validation_prompt": "The video depicts a long, straight highway stretching into the distance, flanked by metal guardrails. The road is divided into multiple lanes, with a few vehicles visible in the far distance. The surrounding landscape features dry, grassy fields on one side and rolling hills on the other. The sky is mostly clear with a few scattered clouds, suggesting a bright, sunny day. And then the camera switch to a inding mountain road covered in snow, with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the solitude and beauty of a winter drive through a mountainous region.",
  "mixed_precision": "bf16",
  "optimizer": "adamw_bf16",
  "learning_rate": 0.00005,
  "max_grad_norm": 0.0,
  "grad_clip_method": "value",
  "lr_scheduler": "cosine",
  "lr_warmup_steps": 400000,
  "base_model_precision": "fp8-torchao",
  "vae_batch_size": 1,
  "webhook_config": "config/ltxvideo/webhook.json",
  "compress_disk_cache": true,
  "use_ema": true,
  "ema_validation": "ema_only",
  "ema_update_interval": 2,
  "delete_problematic_images": "true",
  "disable_bucket_pruning": true,
  "lora_rank": 128,
  "flow_schedule_shift": 1,
  "validation_prompt_library": false,
  "ignore_final_epochs": true
}
```
</details>

### वैकल्पिक: CREPA temporal regularizer

यदि आपके LTX runs में flicker या identity drift दिखे, तो CREPA (cross‑frame alignment) आज़माएँ:
- WebUI में **Training → Loss functions** पर जाएँ और **CREPA** सक्षम करें।
- **Block Index = 8**, **Weight = 0.5**, **Adjacent Distance = 1**, **Temporal Decay = 1.0** से शुरू करें।
- डिफ़ॉल्ट vision encoder (`dinov2_vitg14`, size `518`) रखें। VRAM कम करने की जरूरत हो तो ही `dinov2_vits14` + `224` पर जाएँ।
- पहली बार DINOv2 weights पाने के लिए internet (या cached torch hub) चाहिए।
- वैकल्पिक: यदि आप पूरी तरह cached latents से training कर रहे हैं, तो **Drop VAE Encoder** सक्षम करें; नए videos encode करने की जरूरत हो तो इसे ऑफ रखें।

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

> ℹ️ LTX Video, T5 XXL‑आधारित flow‑matching मॉडल है; छोटे prompts में पर्याप्त जानकारी नहीं हो सकती। लंबे, अधिक वर्णनात्मक prompts उपयोग करें।

#### CLIP score ट्रैकिंग

वर्तमान समय में वीडियो मॉडल प्रशिक्षण के लिए यह सक्षम नहीं किया जाना चाहिए।

</details>

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

`validation_preview_steps` को ऊँचा मान (जैसे 3 या 5) रखें ताकि Tiny AutoEncoder का ओवरहेड कम हो। `validation_num_inference_steps=20` और `validation_preview_steps=5` के साथ, आपको steps 5, 10, 15, और 20 पर preview images मिलेंगी।

#### Flow‑matching schedule shift

Flux, Sana, SD3, और LTX Video जैसे flow‑matching मॉडलों में `shift` नाम का गुण होता है जो timestep schedule के प्रशिक्षित हिस्से को एक सरल decimal value से शिफ्ट करने देता है।

##### Defaults
डिफ़ॉल्ट रूप से कोई schedule shift नहीं लागू होता, जिससे timestep sampling distribution में sigmoid bell‑shape बनती है, जिसे `logit_norm` भी कहते हैं।

##### Auto‑shift
एक सामान्य रूप से अनुशंसित तरीका यह है कि resolution‑dependent timestep shift सक्षम किया जाए, `--flow_schedule_auto_shift`, जो बड़े images के लिए उच्च shift मान और छोटे images के लिए कम shift मान उपयोग करता है। इससे स्थिर लेकिन संभवतः औसत प्रशिक्षण परिणाम मिलते हैं।

##### Manual specification
_Discord के General Awareness का इन उदाहरणों के लिए धन्यवाद_

> ℹ️ ये उदाहरण Flux Dev के साथ value का प्रभाव दिखाते हैं, लेकिन LTX Video भी काफ़ी समान होना चाहिए।

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
    "cache_dir_vae": "cache/vae/ltxvideo/disney-black-and-white",
    "instance_data_dir": "datasets/disney-black-and-white",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 125,
        "min_frames": 125,
        "bucket_strategy": "aspect_ratio"
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/ltxvideo",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> `caption_strategy` विकल्प और आवश्यकताओं के लिए [DATALOADER.md](../DATALOADER.md#caption_strategy) देखें।

- `video` उप‑सेक्शन में हम ये keys सेट कर सकते हैं:
  - `num_frames` (वैकल्पिक, int) बताता है कि प्रशिक्षण में कितने फ्रेम उपयोग होंगे।
    - 25 fps पर 125 फ्रेम 5 सेकंड की वीडियो बनाते हैं, यह standard आउटपुट है। यही आपका target होना चाहिए।
  - `min_frames` (वैकल्पिक, int) प्रशिक्षण के लिए वीडियो की न्यूनतम लंबाई तय करता है।
    - यह कम से कम `num_frames` के बराबर होना चाहिए। इसे न सेट करने पर यह बराबर रखा जाता है।
  - `max_frames` (वैकल्पिक, int) प्रशिक्षण के लिए वीडियो की अधिकतम लंबाई तय करता है।
  - `is_i2v` (वैकल्पिक, bool) बताता है कि डेटासेट पर i2v प्रशिक्षण किया जाएगा या नहीं।
    - यह LTX के लिए डिफ़ॉल्ट रूप से True है। आप चाहें तो इसे बंद कर सकते हैं।
  - `bucket_strategy` (वैकल्पिक, string) वीडियो को buckets में समूहित करने का तरीका तय करता है:
    - `aspect_ratio` (डिफ़ॉल्ट): केवल spatial aspect ratio से समूहित (जैसे `1.78`, `0.75`).
    - `resolution_frames`: `WxH@F` फॉर्मैट (जैसे `768x512@125`) में resolution और फ्रेम काउंट के अनुसार समूहित। mixed‑resolution/duration datasets के लिए उपयोगी।
  - `frame_interval` (वैकल्पिक, int) `resolution_frames` उपयोग करते समय फ्रेम काउंट को इस इंटरवल तक राउंड करता है। इसे अपने मॉडल के आवश्यक फ्रेम‑फैक्टर पर सेट करें।

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

### सबसे कम VRAM कॉन्फ़िग

अन्य मॉडलों की तरह, सबसे कम VRAM उपयोग संभवतः इस कॉन्फ़िग से मिल सकता है:

- OS: Ubuntu Linux 24
- GPU: एक NVIDIA CUDA डिवाइस (10G, 12G)
- System memory: लगभग 11G सिस्टम मेमोरी
- Base model precision: `nf4-bnb`
- Optimiser: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolution: 480px
- Batch size: 1, zero gradient accumulation steps
- DeepSpeed: बंद/कॉन्फ़िगर नहीं
- PyTorch: 2.6
- `--gradient_checkpointing` ज़रूर सक्षम करें, वरना OOM रोकना संभव नहीं होगा

**नोट**: VAE embeds और text encoder outputs की pre‑caching अधिक मेमोरी ले सकती है और फिर भी OOM हो सकता है। ऐसा हो तो text encoder quantisation और VAE tiling सक्षम किया जा सकता है। इन विकल्पों के अलावा, `--offload_during_startup=true` VAE और text encoder मेमोरी उपयोग के बीच प्रतिस्पर्धा से बचने में मदद करेगा।

M3 Max Macbook Pro पर गति लगभग 0.8 iterations per second थी।

### SageAttention

`--attention_mechanism=sageattention` उपयोग करने पर validation समय पर inference तेज़ हो सकता है।

**नोट**: यह हर मॉडल कॉन्फ़िगरेशन के साथ संगत नहीं है, लेकिन कोशिश करने लायक है।

### NF4‑quantised training

सरल शब्दों में, NF4 मॉडल का 4‑bit‑ish प्रतिनिधित्व है, जिसका मतलब है कि training में गंभीर stability चिंताएँ होती हैं।

प्रारंभिक परीक्षणों में:
- Lion optimiser मॉडल collapse करता है लेकिन सबसे कम VRAM उपयोग करता है; AdamW variants मॉडल को संभालने में मदद करते हैं; bnb-adamw8bit, adamw_bf16 अच्छे विकल्प हैं
  - AdEMAMix अच्छा नहीं चला, लेकिन सेटिंग्स नहीं खोजी गईं
- `--max_grad_norm=0.01` मॉडल टूटने को कम करता है क्योंकि यह बहुत बड़े बदलाव रोकता है
- NF4, AdamW8bit, और बड़ा batch size stability समस्याएँ कम करते हैं, लेकिन training समय/VRAM बढ़ता है
- resolution बढ़ाने से training बहुत धीमी हो जाती है और मॉडल को नुकसान हो सकता है
- videos की लंबाई बढ़ाने से मेमोरी बहुत बढ़ती है; `num_frames` घटाएँ
- जो भी int8 या bf16 पर ट्रेन करना कठिन है, वह NF4 में और कठिन होता है
- SageAttention जैसे विकल्पों के साथ कम संगत

NF4 torch.compile के साथ काम नहीं करता, इसलिए जो भी गति मिलेगी वही रहेगी।

यदि VRAM समस्या नहीं है, तो int8 + torch.compile सबसे तेज़ विकल्प है।

### Masked loss

LTX Video के साथ इसका उपयोग न करें।


### Quantisation
- इस मॉडल को ट्रेन करने के लिए quantisation आवश्यक नहीं है

### Image artifacts
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

### Training custom fine‑tuned LTX models

Hugging Face Hub पर कुछ fine‑tuned मॉडलों में पूरा directory structure नहीं होता, जिससे कुछ विकल्प सेट करने पड़ते हैं।

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
    "model_family": "ltxvideo",
    "pretrained_model_name_or_path": "Lightricks/LTX-Video",
    "pretrained_transformer_model_name_or_path": "path/to-the-other-model",
    "pretrained_vae_model_name_or_path": "Lightricks/LTX-Video",
    "pretrained_transformer_subfolder": "none",
}
```
</details>

## Credits

[finetrainers](https://github.com/a-r-r-o-w/finetrainers) प्रोजेक्ट और Diffusers टीम।
- Originally कुछ design concepts SimpleTuner से लिए
- अब video training को आसानी से लागू करने के लिए insight और code योगदान करते हैं
