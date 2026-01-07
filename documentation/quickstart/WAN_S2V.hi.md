## Wan 2.2 S2V त्वरित शुरुआत

इस उदाहरण में हम Wan 2.2 S2V (Speech-to-Video) LoRA को ट्रेन करेंगे। S2V मॉडल ऑडियो इनपुट के आधार पर वीडियो बनाते हैं, जिससे ऑडियो-ड्रिवन वीडियो जनरेशन संभव होती है।

### हार्डवेयर आवश्यकताएं

Wan 2.2 S2V **14B** एक भारी मॉडल है जिसे काफी GPU मेमोरी चाहिए।

#### Speech to Video

14B - https://huggingface.co/tolgacangoz/Wan2.2-S2V-14B-Diffusers
- रिज़ॉल्यूशन: 832x480
- यह 24G में फिट हो जाएगा, लेकिन आपको सेटिंग्स में थोड़ा छेड़छाड़ करनी पड़ेगी।

आपको चाहिए:
- **एक व्यावहारिक न्यूनतम** 24GB या एक सिंगल 4090 या A6000 GPU
- **आदर्श रूप से** कई 4090, A6000, L40S या उससे बेहतर

Apple silicon सिस्टम अभी Wan 2.2 के साथ बहुत अच्छी तरह काम नहीं करते; एक सिंगल ट्रेनिंग स्टेप के लिए लगभग 10 मिनट लग सकते हैं।

### पूर्व-आवश्यकताएँ

सुनिश्चित करें कि आपके पास Python इंस्टॉल है; SimpleTuner 3.10 से 3.12 तक ठीक चलता है।

आप इसे इस कमांड से जांच सकते हैं:

```bash
python --version
```

अगर Ubuntu पर Python 3.12 इंस्टॉल नहीं है, तो आप यह आज़मा सकते हैं:

```bash
apt -y install python3.12 python3.12-venv
```

#### कंटेनर इमेज डिपेंडेंसीज़

Vast, RunPod और TensorDock (आदि) के लिए, CUDA 12.2-12.8 इमेज पर CUDA एक्सटेंशन कम्पाइल करने हेतु यह काम करेगा:

```bash
apt -y install nvidia-cuda-toolkit
```

### इंस्टॉलेशन

pip के जरिए SimpleTuner इंस्टॉल करें:

```bash
pip install simpletuner[cuda]
```

मैनुअल इंस्टॉलेशन या डेवलपमेंट सेटअप के लिए [इंस्टॉलेशन डॉक्यूमेंटेशन](/documentation/INSTALL.md) देखें।
#### SageAttention 2

यदि आप SageAttention 2 का उपयोग करना चाहते हैं, तो कुछ चरणों का पालन करना होगा।

> नोट: SageAttention से बहुत कम स्पीड-अप मिलता है, बहुत प्रभावी नहीं; वजह स्पष्ट नहीं। 4090 पर टेस्ट किया गया।

अपने Python venv के अंदर रहते हुए यह चलाएं:
```bash
git clone https://github.com/thu-ml/SageAttention
pushd SageAttention
  pip install . --no-build-isolation
popd
```

#### AMD ROCm के बाद के चरण

AMD MI300X को उपयोगी बनाने के लिए ये चरण चलाने होंगे:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

### वातावरण सेटअप

SimpleTuner चलाने के लिए आपको एक कॉन्फ़िगरेशन फ़ाइल, डेटासेट और मॉडल डायरेक्टरी, और एक डाटालोडर कॉन्फ़िगरेशन फ़ाइल सेट करनी होगी।

#### कॉन्फ़िगरेशन फ़ाइल

एक प्रयोगात्मक स्क्रिप्ट `configure.py` इंटरैक्टिव चरण-दर-चरण कॉन्फ़िगरेशन के जरिए इस सेक्शन को पूरी तरह स्किप करने में मदद कर सकती है। इसमें कुछ सेफ्टी फीचर हैं जो सामान्य गलतियों से बचाते हैं।

**नोट:** यह आपका डाटालोडर कॉन्फ़िगर नहीं करता। आपको इसे बाद में मैनुअली करना होगा।

इसे चलाने के लिए:

```bash
simpletuner configure
```

> जिन देशों में Hugging Face Hub आसानी से उपलब्ध नहीं है, वहां के उपयोगकर्ताओं को अपने सिस्टम के `$SHELL` के अनुसार `~/.bashrc` या `~/.zshrc` में `HF_ENDPOINT=https://hf-mirror.com` जोड़ना चाहिए।

### मेमोरी ऑफलोडिंग (वैकल्पिक)

Wan, SimpleTuner के सबसे भारी मॉडलों में से एक है। अगर आप VRAM सीमा के करीब हैं तो ग्रुप्ड ऑफलोडिंग चालू करें:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- केवल CUDA डिवाइस `--group_offload_use_stream` को सम्मानित करते हैं; ROCm/MPS अपने आप fallback करते हैं।
- डिस्क स्टेजिंग को कमेंटेड रखें जब तक CPU मेमोरी bottleneck न हो।
- `--enable_model_cpu_offload` ग्रुप ऑफलोडिंग के साथ mutually exclusive है।

### फीड-फॉरवर्ड चंकिंग (वैकल्पिक)

अगर 14B checkpoints ग्रेडिएंट चेकपॉइंटिंग के दौरान अभी भी OOM करते हैं, तो Wan की फीड-फॉरवर्ड लेयर्स को चंक करें:

```bash
--enable_chunked_feed_forward \
--feed_forward_chunk_size 2 \
```

यह कॉन्फ़िगरेशन विज़ार्ड में नए टॉगल (`Training -> Memory Optimisation`) से मेल खाता है। छोटे चंक साइज ज्यादा मेमोरी बचाते हैं लेकिन हर स्टेप को धीमा करते हैं। जल्दी प्रयोगों के लिए आप अपने वातावरण में `WAN_FEED_FORWARD_CHUNK_SIZE=2` भी सेट कर सकते हैं।


अगर आप मैनुअली कॉन्फ़िगर करना पसंद करते हैं:

`config/config.json.example` को `config/config.json` में कॉपी करें:

```bash
cp config/config.json.example config/config.json
```

Multi-GPU उपयोगकर्ता GPU की संख्या कॉन्फ़िगर करने की जानकारी के लिए [इस डॉक्यूमेंट](/documentation/OPTIONS.md#environment-configuration-variables) को देख सकते हैं।

आखिर में आपका कॉन्फ़िगरेशन कुछ ऐसा दिखेगा:

<details>
<summary>View example config</summary>

```json
{
  "resume_from_checkpoint": "latest",
  "quantize_via": "cpu",
  "attention_mechanism": "sageattention",
  "data_backend_config": "config/wan_s2v/multidatabackend.json",
  "aspect_bucket_rounding": 2,
  "seed": 42,
  "minimum_image_size": 0,
  "offload_during_startup": true,
  "disable_benchmark": false,
  "output_dir": "output/wan_s2v",
  "lora_type": "standard",
  "lycoris_config": "config/wan_s2v/lycoris_config.json",
  "max_train_steps": 400000,
  "num_train_epochs": 0,
  "checkpoint_step_interval": 1000,
  "checkpoints_total_limit": 5,
  "hub_model_id": "wan-s2v-lora",
  "push_to_hub": "true",
  "push_checkpoints_to_hub": "true",
  "tracker_project_name": "lora-training",
  "tracker_run_name": "wan-s2v-adamW",
  "report_to": "wandb",
  "model_type": "lora",
  "pretrained_model_name_or_path": "tolgacangoz/Wan2.2-S2V-14B-Diffusers",
  "pretrained_t5_model_name_or_path": "tolgacangoz/Wan2.2-S2V-14B-Diffusers",
  "model_family": "wan_s2v",
  "model_flavour": "s2v-14b-2.2",
  "train_batch_size": 2,
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 1,
  "caption_dropout_probability": 0.1,
  "resolution_type": "pixel_area",
  "resolution": 480,
  "validation_seed": 42,
  "validation_step_interval": 100,
  "validation_resolution": "832x480",
  "validation_prompt": "A person speaking with natural gestures",
  "validation_negative_prompt": "blurry, low quality, distorted",
  "validation_guidance": 4.5,
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
  "webhook_config": "config/wan_s2v/webhook.json",
  "compress_disk_cache": true,
  "use_ema": true,
  "ema_validation": "ema_only",
  "ema_update_interval": 2,
  "delete_problematic_images": "true",
  "disable_bucket_pruning": true,
  "lora_rank": 16,
  "flow_schedule_shift": 3,
  "validation_prompt_library": false,
  "ignore_final_epochs": true
}
```
</details>

इस कॉन्फ़िगरेशन में विशेष रूप से महत्वपूर्ण वैलिडेशन सेटिंग्स हैं। इनके बिना आउटपुट बहुत अच्छे नहीं दिखते।

### वैकल्पिक: CREPA टेम्पोरल रेग्युलराइज़र

Wan S2V में स्मूद मूवमेंट और कम आइडेंटिटी ड्रिफ्ट के लिए:
- **Training -> Loss functions** में **CREPA** सक्षम करें।
- **Block Index = 8**, **Weight = 0.5**, **Adjacent Distance = 1**, **Temporal Decay = 1.0** से शुरू करें।
- डिफ़ॉल्ट एन्कोडर (`dinov2_vitg14`, साइज `518`) अच्छी तरह काम करता है; केवल VRAM बचाने की जरूरत हो तो `dinov2_vits14` + `224` पर जाएं।
- पहली रन में torch hub के जरिए DINOv2 डाउनलोड होता है; अगर आप ऑफ़लाइन ट्रेन करते हैं तो इसे कैश या प्रीफेच करें।
- **Drop VAE Encoder** तभी सक्षम करें जब आप पूरी तरह कैश्ड लेटेंट्स से ट्रेन कर रहे हों; नहीं तो इसे ऑफ रखें ताकि पिक्सेल एन्कोड सही से काम करें।

### उन्नत प्रयोगात्मक सुविधाएँ

<details>
<summary>Show advanced experimental details</summary>


SimpleTuner में प्रयोगात्मक फीचर हैं जो ट्रेनिंग की स्थिरता और प्रदर्शन को काफी बेहतर कर सकते हैं।

*   **[Scheduled Sampling (Rollout)](/documentation/experimental/SCHEDULED_SAMPLING.md):** एक्सपोज़र बायस घटाता है और आउटपुट क्वालिटी बेहतर करता है, क्योंकि ट्रेनिंग के दौरान मॉडल को अपने इनपुट्स खुद जनरेट करने देता है।

> ये फीचर ट्रेनिंग का कंप्यूटेशनल ओवरहेड बढ़ाते हैं।

</details>

### TREAD ट्रेनिंग

> **Experimental**: TREAD नया लागू किया गया फीचर है। यह काम करता है, लेकिन सबसे अच्छे कॉन्फ़िगरेशन अभी भी खोजे जा रहे हैं।

[TREAD](/documentation/TREAD.md) (paper) का मतलब है **T**oken **R**outing for **E**fficient **A**rchitecture-agnostic **D**iffusion। यह तरीका Wan S2V ट्रेनिंग को तेज कर सकता है क्योंकि यह ट्रांसफॉर्मर लेयर्स के बीच टोकन को समझदारी से रूट करता है। स्पीडअप इस बात पर निर्भर है कि आप कितने टोकन ड्रॉप करते हैं।

#### त्वरित सेटअप

सरल और कंज़र्वेटिव अप्रोच के लिए इसे अपने `config.json` में जोड़ें:

<details>
<summary>View example config</summary>

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

यह कॉन्फ़िगरेशन:
- लेयर 2 से लेकर दूसरी-आखिरी लेयर तक केवल 50% इमेज टोकन रखेगा
- टेक्स्ट टोकन कभी ड्रॉप नहीं होते
- न्यूनतम क्वालिटी प्रभाव के साथ ~25% ट्रेनिंग स्पीडअप
- ट्रेनिंग क्वालिटी और कन्वर्जेंस में संभावित सुधार

#### मुख्य बिंदु

- **सीमित आर्किटेक्चर सपोर्ट** - TREAD केवल Flux और Wan मॉडलों (S2V सहित) के लिए लागू है
- **उच्च रिज़ॉल्यूशन पर बेहतर** - 1024x1024+ पर सबसे बड़े स्पीडअप, क्योंकि attention की जटिलता O(n^2) है
- **masked loss के साथ संगत** - masked क्षेत्रों को अपने आप बचाया जाता है (लेकिन इससे स्पीडअप कम होता है)
- **quantization के साथ काम करता है** - int8/int4/NF4 ट्रेनिंग के साथ जोड़ा जा सकता है
- **शुरुआती loss spike की उम्मीद करें** - LoRA/LoKr ट्रेनिंग शुरू करते समय loss शुरू में ज्यादा होगा लेकिन जल्दी ठीक हो जाता है

#### ट्यूनिंग टिप्स

- **कंज़र्वेटिव (क्वालिटी फोकस)**: `selection_ratio` को 0.1-0.3 रखें
- **एग्रेसिव (स्पीड फोकस)**: `selection_ratio` को 0.3-0.5 रखें और क्वालिटी प्रभाव स्वीकार करें
- **शुरुआती/अंतिम लेयर से बचें**: लेयर 0-1 या अंतिम लेयर में रूटिंग न करें
- **LoRA ट्रेनिंग के लिए**: हल्की धीमी गति दिख सकती है - अलग-अलग कॉन्फ़िग ट्राई करें
- **ऊंचा रिज़ॉल्यूशन = बेहतर स्पीडअप**: 1024px और उससे ऊपर में अधिक लाभ

डिटेल्ड कॉन्फ़िगरेशन विकल्प और ट्रबलशूटिंग के लिए [TREAD की पूरी डॉक्यूमेंटेशन](/documentation/TREAD.md) देखें।


#### वैधता प्रॉम्प्ट

`config/config.json` के अंदर "primary validation prompt" होता है, जो आम तौर पर आपका मुख्य instance_prompt होता है जिसे आप अपने सिंगल सब्जेक्ट या स्टाइल के लिए ट्रेन कर रहे हैं। इसके अलावा, एक JSON फ़ाइल बनाई जा सकती है जिसमें अतिरिक्त प्रॉम्प्ट हों जिन्हें वैलिडेशन के दौरान चलाया जाए।

उदाहरण कॉन्फ़िग फ़ाइल `config/user_prompt_library.json.example` में यह फॉर्मेट है:

<details>
<summary>View example config</summary>

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```
</details>

ये nicknames वैलिडेशन के लिए फ़ाइलनाम होते हैं, इसलिए इन्हें छोटा और आपके फ़ाइल सिस्टम के अनुकूल रखें।

इस प्रॉम्प्ट लाइब्रेरी को ट्रेनर में जोड़ने के लिए, `config.json` के अंत में नई लाइन जोड़कर इसे TRAINER_EXTRA_ARGS में डालें:
<details>
<summary>View example config</summary>

```json
  "--user_prompt_library": "config/user_prompt_library.json",
```
</details>

> S2V UMT5 टेक्स्ट एन्कोडर का उपयोग करता है, जिसमें embeddings में काफी लोकल जानकारी होती है, इसलिए छोटे प्रॉम्प्ट पर्याप्त जानकारी नहीं दे पाते। बेहतर परिणाम के लिए लंबे और अधिक वर्णनात्मक प्रॉम्प्ट इस्तेमाल करें।

#### CLIP स्कोर ट्रैकिंग

यह वीडियो मॉडल ट्रेनिंग के लिए फिलहाल सक्षम नहीं करना चाहिए।

# स्थिर मूल्यांकन लॉस

अगर आप मॉडल के प्रदर्शन को स्कोर करने के लिए stable MSE loss का उपयोग करना चाहते हैं, तो कॉन्फ़िगरेशन और व्याख्या के लिए [यह दस्तावेज़](/documentation/evaluation/EVAL_LOSS.md) देखें।

#### वैधता प्रीव्यू

SimpleTuner जेनरेशन के दौरान Tiny AutoEncoder मॉडल का उपयोग करके स्ट्रीमिंग इंटरमीडिएट वैलिडेशन प्रीव्यू सपोर्ट करता है। इससे आप वेबहुक कॉलबैक के जरिए रियल-टाइम में वैलिडेशन इमेजेस को स्टेप-बाय-स्टेप देख सकते हैं।

इसे सक्षम करने के लिए:
<details>
<summary>View example config</summary>

```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```
</details>

**आवश्यकताएँ:**
- वेबहुक कॉन्फ़िगरेशन
- वैलिडेशन सक्षम होना चाहिए

Tiny AutoEncoder ओवरहेड घटाने के लिए `validation_preview_steps` को ज्यादा (जैसे 3 या 5) सेट करें। `validation_num_inference_steps=20` और `validation_preview_steps=5` के साथ आपको स्टेप 5, 10, 15, और 20 पर प्रीव्यू इमेजेस मिलेंगी।

#### फ्लो-मैचिंग शेड्यूल शिफ्ट

Flux, Sana, SD3, LTX Video और Wan S2V जैसे flow-matching मॉडल्स में `shift` नाम की एक प्रॉपर्टी होती है, जो हमें किसी सरल दशमलव मान के जरिए टाइमस्टेप शेड्यूल के प्रशिक्षित हिस्से को शिफ्ट करने देती है।

##### डिफ़ॉल्ट
डिफ़ॉल्ट रूप से कोई शेड्यूल शिफ्ट लागू नहीं होता, जिससे टाइमस्टेप सैम्पलिंग वितरण में सिग्मॉइड बेल-शेप बनता है, जिसे `logit_norm` कहा जाता है।

##### ऑटो-शिफ्ट
सामान्य तौर पर सुझाया गया तरीका यह है कि कई हालिया कार्यों का अनुसरण करते हुए रेज़ॉल्यूशन-डिपेंडेंट टाइमस्टेप शिफ्ट, `--flow_schedule_auto_shift` को सक्षम करें, जो बड़े इमेज के लिए बड़े शिफ्ट और छोटे इमेज के लिए छोटे शिफ्ट का उपयोग करता है। इससे स्थिर लेकिन संभवतः औसत परिणाम मिलते हैं।

##### मैनुअल स्पेसिफिकेशन
_Discord पर General Awareness को इन उदाहरणों के लिए धन्यवाद_

> ये उदाहरण Flux Dev का उपयोग करके मान का प्रभाव दिखाते हैं, हालांकि Wan S2V बहुत समान होना चाहिए।

`--flow_schedule_shift` का मान 0.1 (बहुत कम) होने पर, केवल इमेज के फाइन डिटेल्स प्रभावित होते हैं:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

`--flow_schedule_shift` का मान 4.0 (बहुत अधिक) होने पर, बड़े कॉम्पोज़िशनल फीचर्स और संभवतः मॉडल का कलर स्पेस प्रभावित होता है:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)


#### क्वांटाइज़्ड मॉडल ट्रेनिंग

Apple और NVIDIA सिस्टम्स पर टेस्ट किया गया; Hugging Face Optimum-Quanto का उपयोग प्रिसिजन और VRAM आवश्यकताओं को घटाने के लिए किया जा सकता है, जिससे केवल 16GB में ट्रेनिंग संभव है।



`config.json` उपयोगकर्ताओं के लिए:
<details>
<summary>View example config</summary>

```json
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "lora_rank": 16,
  "max_grad_norm": 1.0,
  "base_model_default_dtype": "bf16"
```
</details>

#### वैधता सेटिंग्स

प्रारंभिक एक्सप्लोरेशन के दौरान खराब आउटपुट क्वालिटी Wan S2V की वजह से हो सकती है, और यह कुछ कारणों पर निर्भर करता है:

- इंफरेंस के लिए पर्याप्त स्टेप्स नहीं
  - जब तक आप UniPC नहीं इस्तेमाल कर रहे, आपको शायद कम से कम 40 स्टेप्स चाहिए। UniPC संख्या थोड़ी कम कर सकता है, लेकिन आपको प्रयोग करना होगा।
- scheduler कॉन्फ़िगरेशन गलत
  - यह सामान्य Euler flow matching शेड्यूल का उपयोग कर रहा था, लेकिन Betas वितरण सबसे अच्छा लगता है
  - यदि आपने इस सेटिंग को नहीं बदला है, तो यह अब ठीक होना चाहिए
- रिज़ॉल्यूशन गलत
  - Wan S2V केवल उन्हीं रिज़ॉल्यूशनों पर सही काम करता है जिन पर इसे ट्रेन किया गया था; अगर यह काम कर जाए तो यह किस्मत है, लेकिन खराब परिणाम सामान्य हैं
- CFG वैल्यू गलत
  - 4.0-5.0 के आसपास का मान सुरक्षित लगता है
- खराब prompting
  - निश्चित रूप से, वीडियो मॉडल्स को ऐसे मिस्टिक लोगों की टीम चाहिए जो महीनों तक पहाड़ों में ज़ेन रिट्रीट पर जाएं ताकि prompting की पवित्र कला सीख सकें, क्योंकि इनके डेटासेट और कैप्शन स्टाइल पवित्र ग्रेल की तरह संरक्षित हैं।
  - tl;dr अलग-अलग प्रॉम्प्ट आज़माएं।
- ऑडियो अनुपस्थित या मिसमैच
  - S2V को वैलिडेशन के लिए ऑडियो इनपुट चाहिए - सुनिश्चित करें कि आपके वैलिडेशन सैंपल्स के साथ संबंधित ऑडियो फाइलें हों

इसके बावजूद, जब तक आपका बैच साइज बहुत कम और/या लर्निंग रेट बहुत अधिक नहीं है, मॉडल आपके पसंदीदा इंफरेंस टूल में ठीक से चलेगा (मानते हुए कि आपके पास पहले से एक है जो अच्छे परिणाम देता है)।

#### डेटासेट संबंधी विचार

S2V ट्रेनिंग के लिए वीडियो और ऑडियो डेटा जोड़े में होना चाहिए। डिफ़ॉल्ट रूप से SimpleTuner वीडियो डेटासेट से ऑडियो
ऑटो-स्प्लिट करता है, इसलिए अलग ऑडियो डेटासेट तभी चाहिए जब आप कस्टम प्रोसेसिंग चाहते हों। `audio.auto_split: false`
से opt-out करें और `s2v_datasets` मैन्युअली दें।

डेटासेट साइज पर बहुत कम सीमाएं हैं, सिवाय इसके कि इसे प्रोसेस और ट्रेन करने में कितना कंप्यूट और समय लगेगा।

आपको सुनिश्चित करना होगा कि डेटासेट इतना बड़ा हो कि आपका मॉडल प्रभावी ढंग से ट्रेन हो सके, लेकिन इतना बड़ा भी न हो कि उपलब्ध कंप्यूट से बाहर हो जाए।

ध्यान दें कि न्यूनतम डेटासेट साइज `train_batch_size * gradient_accumulation_steps` के बराबर और `vae_batch_size` से बड़ा होना चाहिए। अगर यह बहुत छोटा है तो डेटासेट उपयोगी नहीं होगा।

> पर्याप्त कम सैंपल होने पर आपको संदेश **no samples detected in dataset** दिख सकता है - `repeats` वैल्यू बढ़ाने से यह सीमा दूर हो जाएगी।

#### ऑडियो डेटासेट सेटअप

##### वीडियो से स्वचालित ऑडियो निकासी (अनुशंसित)

अगर आपके वीडियो में पहले से ऑडियो ट्रैक हैं, तो SimpleTuner अलग ऑडियो डेटासेट की जरूरत के बिना ऑडियो स्वतः एक्सट्रैक्ट और प्रोसेस कर सकता है। यह सबसे सरल और डिफ़ॉल्ट तरीका है:

```json
[
  {
    "id": "s2v-videos",
    "type": "local",
    "dataset_type": "video",
    "crop": false,
    "resolution": 480,
    "minimum_image_size": 480,
    "maximum_image_size": 480,
    "target_downsample_size": 480,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/wan_s2v/videos",
    "instance_data_dir": "datasets/s2v-videos",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 75,
        "min_frames": 75,
        "bucket_strategy": "aspect_ratio"
    },
    "audio": {
        "auto_split": true,
        "sample_rate": 16000,
        "channels": 1
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/wan_s2v",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

ऑडियो auto-split सक्षम होने पर (डिफ़ॉल्ट), SimpleTuner:
1. ऑटो-जनरेटेड ऑडियो डेटासेट कॉन्फ़िगरेशन बनाता है (`s2v-videos_audio`)
2. मेटाडेटा डिस्कवरी के दौरान हर वीडियो से ऑडियो निकालता है
3. ऑडियो VAE लेटेंट्स को एक अलग डायरेक्टरी में कैश करता है
4. `s2v_datasets` के जरिए ऑडियो डेटासेट को अपने आप लिंक करता है

**ऑडियो कॉन्फ़िगरेशन विकल्प:**
- `audio.auto_split` (bool): वीडियो से ऑडियो की ऑटोमैटिक एक्सट्रैक्शन सक्षम करें (डिफ़ॉल्ट: true)
- `audio.sample_rate` (int): लक्ष्य सैंपल रेट Hz में (डिफ़ॉल्ट: Wav2Vec2 के लिए 16000)
- `audio.channels` (int): ऑडियो चैनलों की संख्या (डिफ़ॉल्ट: 1 मोनो के लिए)
- `audio.allow_zero_audio` (bool): बिना ऑडियो स्ट्रीम वाले वीडियो के लिए ज़ीरो-फिल्ड ऑडियो बनाएं (डिफ़ॉल्ट: false)
- `audio.max_duration_seconds` (float): अधिकतम ऑडियो अवधि; इससे लंबे फाइलें स्किप होंगी
- `audio.duration_interval` (float): सेकंड में बकेट ग्रुपिंग के लिए अवधि अंतराल (डिफ़ॉल्ट: 3.0)
- `audio.truncation_mode` (string): लंबे ऑडियो को कैसे ट्रंकेट करें: "beginning", "end", "random" (डिफ़ॉल्ट: "beginning")

**नोट**: जिन वीडियो में ऑडियो ट्रैक नहीं हैं, वे S2V ट्रेनिंग के लिए स्वतः स्किप हो जाते हैं, जब तक `audio.allow_zero_audio: true` सेट न हो।

##### मैनुअल ऑडियो डेटासेट (वैकल्पिक)

अगर आप अलग ऑडियो फाइलें रखना चाहते हैं, कस्टम ऑडियो प्रोसेसिंग चाहिए या auto-split बंद करते हैं, तो S2V मॉडल
प्री-एक्सट्रैक्टेड ऑडियो फाइलों का भी उपयोग कर सकते हैं जो आपके वीडियो फाइलों से नाम के आधार पर मेल खाती हों। उदाहरण के लिए:
- `video_001.mp4` के साथ संबंधित `video_001.wav` (या `.mp3`, `.flac`, `.ogg`, `.m4a`) होना चाहिए

ऑडियो फाइलें एक अलग डायरेक्टरी में होनी चाहिए जिसे आप `s2v_datasets` बैकएंड के रूप में कॉन्फ़िगर करेंगे।

##### वीडियो से ऑडियो निकालना (मैनुअल)

अगर आपके वीडियो में पहले से ऑडियो है, तो उसे निकालने के लिए दिए गए स्क्रिप्ट का उपयोग करें:

```bash
# Extract audio only (keeps original videos unchanged)
python scripts/generate_s2v_audio.py \
    --input-dir datasets/s2v-videos \
    --output-dir datasets/s2v-audio

# Extract audio and remove it from source videos (recommended to avoid redundant data)
python scripts/generate_s2v_audio.py \
    --input-dir datasets/s2v-videos \
    --output-dir datasets/s2v-audio \
    --strip-audio
```

स्क्रिप्ट:
- 16kHz मोनो WAV में ऑडियो निकालता है (Wav2Vec2 की native sample rate)
- फाइलनाम अपने आप मैच करता है (उदा., `video.mp4` -> `video.wav`)
- जिन वीडियो में ऑडियो स्ट्रीम नहीं है उन्हें स्किप करता है
- `ffmpeg` इंस्टॉल होना चाहिए

##### डेटासेट कॉन्फ़िगरेशन (मैनुअल)

यह वाला `--data_backend_config` (`config/multidatabackend.json`) दस्तावेज़ बनाएं:

<details>
<summary>View example config</summary>

```json
[
  {
    "id": "s2v-videos",
    "type": "local",
    "dataset_type": "video",
    "crop": false,
    "resolution": 480,
    "minimum_image_size": 480,
    "maximum_image_size": 480,
    "target_downsample_size": 480,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/wan_s2v/videos",
    "instance_data_dir": "datasets/s2v-videos",
    "s2v_datasets": ["s2v-audio"],
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
    "id": "s2v-audio",
    "type": "local",
    "dataset_type": "audio",
    "instance_data_dir": "datasets/s2v-audio",
    "disabled": false
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/wan_s2v",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> caption_strategy के विकल्प और आवश्यकताएं [DATALOADER.md](../DATALOADER.md#caption_strategy) में देखें।

S2V डेटासेट कॉन्फ़िगरेशन के मुख्य बिंदु:
- आपके वीडियो डेटासेट पर `s2v_datasets` फ़ील्ड ऑडियो बैकएंड(s) की ओर इशारा करता है
- ऑडियो फाइलें फ़ाइलनाम स्टेम से मैच होती हैं (उदा., `video_001.mp4` का मेल `video_001.wav` से)
- ऑडियो Wav2Vec2 से ऑन-द-फ्लाई एन्कोड होता है (~600MB VRAM), कैशिंग की जरूरत नहीं
- ऑडियो डेटासेट टाइप `audio` है

- `video` सब-सेक्शन में, हम ये कीज़ सेट कर सकते हैं:
  - `num_frames` (वैकल्पिक, int) ट्रेनिंग के लिए फ्रेम्स की संख्या है।
    - 15 fps पर, 75 फ्रेम 5 सेकंड का वीडियो होता है, मानक आउटपुट। यही आपका लक्ष्य होना चाहिए।
  - `min_frames` (वैकल्पिक, int) न्यूनतम वीडियो लंबाई तय करता है जिसे ट्रेनिंग के लिए माना जाएगा।
    - यह कम से कम `num_frames` के बराबर होना चाहिए। इसे सेट न करने पर यह बराबर सुनिश्चित किया जाता है।
  - `max_frames` (वैकल्पिक, int) अधिकतम वीडियो लंबाई तय करता है जिसे ट्रेनिंग के लिए माना जाएगा।
  - `bucket_strategy` (वैकल्पिक, string) वीडियो को बकेट्स में कैसे ग्रुप किया जाए यह तय करता है:
    - `aspect_ratio` (डिफ़ॉल्ट): केवल स्पेशियल आस्पेक्ट रेशियो के हिसाब से ग्रुप (उदा., `1.78`, `0.75`)।
    - `resolution_frames`: `WxH@F` फॉर्मेट में रिज़ॉल्यूशन और फ्रेम काउंट के आधार पर ग्रुप (उदा., `832x480@75`)। मिक्स्ड रिज़ॉल्यूशन/ड्यूरेशन डेटासेट के लिए उपयोगी।
  - `frame_interval` (वैकल्पिक, int) `resolution_frames` उपयोग करते समय फ्रेम काउंट को इस अंतराल पर राउंड करता है।

इसके बाद, अपने वीडियो और ऑडियो फाइलों के साथ `datasets` डायरेक्टरी बनाएं:

```bash
mkdir -p datasets/s2v-videos datasets/s2v-audio
# Place your video files in datasets/s2v-videos/
# Place your audio files in datasets/s2v-audio/
```

हर वीडियो के लिए फ़ाइलनाम स्टेम के आधार पर मिलती-जुलती ऑडियो फ़ाइल होना सुनिश्चित करें।

#### WandB और Huggingface Hub में लॉगिन करें

ट्रेनिंग शुरू करने से पहले WandB और HF Hub में लॉगिन करना बेहतर है, खासकर अगर आप `--push_to_hub` और `--report_to=wandb` उपयोग कर रहे हैं।

अगर आप Git LFS रिपॉजिटरी में चीजें मैनुअली पुश करेंगे, तो `git config --global credential.helper store` भी चलाएं

निम्न कमांड चलाएं:

```bash
wandb login
```

और

```bash
huggingface-cli login
```

दोनों सेवाओं में लॉगिन करने के निर्देशों का पालन करें।

### ट्रेनिंग रन शुरू करना

SimpleTuner डायरेक्टरी से, ट्रेनिंग शुरू करने के लिए आपके पास कई विकल्प हैं:

**विकल्प 1 (अनुशंसित - pip इंस्टॉल):**
```bash
pip install simpletuner[cuda]
simpletuner train
```

**विकल्प 2 (Git clone तरीका):**
```bash
simpletuner train
```

**विकल्प 3 (Legacy तरीका - अभी भी काम करता है):**
```bash
./train.sh
```

इससे text embed और VAE आउटपुट का डिस्क पर कैशिंग शुरू हो जाएगा।

अधिक जानकारी के लिए [dataloader](/documentation/DATALOADER.md) और [tutorial](/documentation/TUTORIAL.md) दस्तावेज़ देखें।

## नोट्स और ट्रबलशूटिंग टिप्स

### न्यूनतम VRAM कॉन्फ़िग

Wan S2V quantization के प्रति संवेदनशील है और अभी NF4 या INT4 के साथ उपयोग नहीं किया जा सकता।

- OS: Ubuntu Linux 24
- GPU: एक सिंगल NVIDIA CUDA डिवाइस (24G अनुशंसित)
- सिस्टम मेमोरी: लगभग 16G सिस्टम मेमोरी
- बेस मॉडल प्रिसिजन: `int8-quanto`
- ऑप्टिमाइज़र: Lion 8Bit Paged, `bnb-lion8bit-paged`
- रिज़ॉल्यूशन: 480px
- बैच साइज: 1, ग्रेडिएंट accumulation के शून्य स्टेप्स
- DeepSpeed: disabled / unconfigured
- PyTorch: 2.6
- `--gradient_checkpointing` जरूर सक्षम करें, वरना कुछ भी OOM रोक नहीं पाएगा
- केवल इमेज पर ट्रेन करें, या अपने वीडियो डेटासेट के लिए `num_frames` को 1 सेट करें

**नोट**: VAE embeds और टेक्स्ट एन्कोडर आउटपुट को pre-cache करने में ज्यादा मेमोरी लग सकती है और फिर भी OOM हो सकता है। इसलिए `--offload_during_startup=true` लगभग अनिवार्य है। ऐसा होने पर टेक्स्ट एन्कोडर quantization और VAE tiling सक्षम की जा सकती है। (Wan अभी VAE tiling/slicing सपोर्ट नहीं करता)

### SageAttention

`--attention_mechanism=sageattention` उपयोग करने पर वैलिडेशन के समय inference तेज हो सकता है।

**नोट**: यह अंतिम VAE decode स्टेप के साथ संगत नहीं है और उस हिस्से को तेज नहीं करेगा।

### Masked loss

इसे Wan S2V के साथ उपयोग न करें।

### Quantisation
- बैच साइज पर निर्भर करते हुए 24G में इस मॉडल को ट्रेन करने के लिए quantization की जरूरत पड़ सकती है

### इमेज आर्टिफैक्ट्स
Wan को Euler Betas flow-matching शेड्यूल या (डिफ़ॉल्ट रूप से) UniPC multistep solver की जरूरत होती है, जो एक higher-order scheduler है और मजबूत प्रेडिक्शन देगा।

अन्य DiT मॉडलों की तरह, अगर आप ये काम करते हैं (और भी कई): कुछ स्क्वेयर ग्रिड आर्टिफैक्ट्स दिखाई दे सकते हैं:
- कम गुणवत्ता वाले डेटा के साथ ओवरट्रेन
- बहुत ऊँची learning rate उपयोग करना
- ओवरट्रेनिंग (सामान्य रूप से), बहुत सी इमेज के साथ low-capacity नेटवर्क
- अंडरट्रेनिंग (भी), बहुत कम इमेज के साथ high-capacity नेटवर्क
- अजीब आस्पेक्ट रेशियो या ट्रेनिंग डेटा साइज का उपयोग

### आस्पेक्ट बकेटिंग
- वीडियो को इमेज की तरह बकेट किया जाता है।
- स्क्वेयर क्रॉप्स पर बहुत लंबे समय तक ट्रेनिंग इस मॉडल को ज्यादा नुकसान नहीं पहुंचाएगी। मज़े करो, यह भरोसेमंद है।
- दूसरी तरफ, आपके डेटासेट के नैचुरल आस्पेक्ट बकेट्स का उपयोग inference समय में इन शेप्स को ज़्यादा bias कर सकता है।
  - यह एक वांछनीय गुण हो सकता है, क्योंकि यह cinematic जैसी आस्पेक्ट-डिपेंडेंट स्टाइल्स को अन्य रिज़ॉल्यूशन में फैलने से रोकता है।
  - हालांकि, अगर आप कई आस्पेक्ट बकेट्स में समान रूप से बेहतर परिणाम चाहते हैं, तो आपको `crop_aspect=random` के साथ प्रयोग करना पड़ सकता है, जो अपनी कमियाँ लाता है।
- इमेज डायरेक्टरी डेटासेट को कई बार परिभाषित करके डेटासेट कॉन्फ़िगरेशन को मिक्स करने से अच्छे परिणाम और अच्छी तरह जनरलाइज़्ड मॉडल मिला है।

### ऑडियो सिंक्रोनाइज़ेशन

S2V के बेहतर परिणामों के लिए:
- सुनिश्चित करें कि ऑडियो की अवधि वीडियो की अवधि से मेल खाती हो
- ऑडियो को अंदरूनी तौर पर 16kHz पर resample किया जाता है
- Wav2Vec2 एन्कोडर ऑन-द-फ्लाई ऑडियो प्रोसेस करता है (~600MB VRAM ओवरहेड)
- ऑडियो फीचर्स को वीडियो फ्रेम्स की संख्या से मैच करने के लिए इंटरपोलेट किया जाता है
