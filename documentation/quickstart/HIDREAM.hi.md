## HiDream क्विकस्टार्ट

इस उदाहरण में हम HiDream के लिए एक Lycoris LoKr ट्रेन करेंगे, उम्मीद है कि हमारे पास इसे चलाने के लिए पर्याप्त मेमोरी होगी।

24G GPU शायद न्यूनतम है जिसे आप व्यापक ब्लॉक ऑफलोडिंग और फ्यूज़्ड बैकवर्ड पास के बिना चला पाएंगे। एक Lycoris LoKr भी उतना ही अच्छा काम करेगा!

### हार्डवेयर आवश्यकताएँ

HiDream कुल 17B पैरामीटर का है, और किसी भी समय लगभग ~8B एक्टिव रहते हैं, जो एक सीखा हुआ MoE गेट उपयोग करके काम को वितरित करता है। यह **चार** टेक्स्ट एन्कोडर और Flux VAE का उपयोग करता है।

कुल मिलाकर मॉडल आर्किटेक्चरल जटिलता से प्रभावित है, और लगता है कि यह Flux Dev का एक डेरिवेटिव है, या तो सीधे डिस्टिलेशन से या निरंतर फाइन-ट्यूनिंग से, जैसा कि कुछ वैलिडेशन सैंपल्स में समान वेट्स दिखते हैं।

### प्रीरिक्विज़िट्स

सुनिश्चित करें कि आपने Python इंस्टॉल कर रखा है; SimpleTuner 3.10 से 3.12 तक अच्छा काम करता है।

आप इसे चलाकर जांच सकते हैं:

```bash
python --version
```

यदि Ubuntu पर Python 3.12 इंस्टॉल नहीं है, तो आप यह आज़मा सकते हैं:

```bash
apt -y install python3.12 python3.12-venv
```

#### कंटेनर इमेज निर्भरताएँ

Vast, RunPod, और TensorDock (और अन्य) के लिए, CUDA 12.2-12.8 इमेज पर CUDA एक्सटेंशन कम्पाइल करने हेतु यह काम करेगा:

```bash
apt -y install nvidia-cuda-toolkit
```

### इंस्टॉलेशन

pip के जरिए SimpleTuner इंस्टॉल करें:

```bash
pip install simpletuner[cuda]
```

मैनुअल इंस्टॉलेशन या डेवलपमेंट सेटअप के लिए [इंस्टॉलेशन डॉक्यूमेंटेशन](../INSTALL.md) देखें।

### एनवायरनमेंट सेटअप करना

SimpleTuner चलाने के लिए आपको एक कॉन्फ़िगरेशन फ़ाइल, डेटासेट और मॉडल डायरेक्टरीज़, और एक डाटालोडर कॉन्फ़िगरेशन फ़ाइल सेट करनी होगी।

#### कॉन्फ़िगरेशन फ़ाइल

एक प्रयोगात्मक स्क्रिप्ट, `configure.py`, इंटरएक्टिव स्टेप-बाय-स्टेप कॉन्फ़िगरेशन के जरिए इस सेक्शन को पूरी तरह स्किप करने दे सकती है। इसमें कुछ सेफ्टी फीचर्स हैं जो सामान्य गलतियों से बचाते हैं।

**नोट:** यह आपके डाटालोडर को कॉन्फ़िगर नहीं करता। वह आपको बाद में मैनुअली करना होगा।

इसे चलाने के लिए:

```bash
simpletuner configure
```

> ⚠️ जिन देशों में Hugging Face Hub आसानी से उपलब्ध नहीं है, वहाँ के उपयोगकर्ताओं को अपने `~/.bashrc` या `~/.zshrc` में `HF_ENDPOINT=https://hf-mirror.com` जोड़ना चाहिए, आपके सिस्टम का `$SHELL` जो भी उपयोग कर रहा हो उसके अनुसार।


अगर आप मैनुअली कॉन्फ़िगर करना चाहते हैं:

`config/config.json.example` को `config/config.json` में कॉपी करें:

```bash
cp config/config.json.example config/config.json
```

वहाँ आपको संभवतः निम्न वेरिएबल्स बदलने होंगे:

- `model_type` - इसे `lora` सेट करें।
- `lora_type` - इसे `lycoris` सेट करें।
- `model_family` - इसे `hidream` सेट करें।
- `model_flavour` - इसे `full` सेट करें, क्योंकि `dev` इस तरह डिस्टिल्ड है कि यदि आप डिस्टिलेशन तोड़ने तक नहीं जाना चाहते तो इसे सीधे ट्रेन करना आसान नहीं है।
  - वास्तव में, `full` मॉडल भी ट्रेन करना कठिन है, लेकिन वही एक मॉडल है जिसे डिस्टिल नहीं किया गया है।
- `output_dir` - इसे उस डायरेक्टरी पर सेट करें जहाँ आप अपने चेकपॉइंट्स और वैलिडेशन इमेजेस रखना चाहते हैं। यहाँ पूर्ण पाथ उपयोग करना बेहतर है।
- `train_batch_size` - 1, शायद?
- `validation_resolution` - इसे `1024x1024` या HiDream के अन्य समर्थित रेजोल्यूशन में सेट करें।
  - अन्य रेजोल्यूशन कॉमा से अलग करके दे सकते हैं: `1024x1024,1280x768,2048x2048`
- `validation_guidance` - HiDream के लिए जो आप इंफरेंस में चुनते हैं वही इस्तेमाल करें; 2.5-3.0 के आस-पास का कम वैल्यू अधिक वास्तविक परिणाम देता है
- `validation_num_inference_steps` - लगभग 30 के आसपास रखें
- `use_ema` - इसे `true` करने से आपके मुख्य ट्रेन किए गए चेकपॉइंट के साथ एक ज्यादा स्मूद परिणाम मिलता है।

- `optimizer` - आप कोई भी ऑप्टिमाइज़र इस्तेमाल कर सकते हैं जिसमें आप सहज हों, लेकिन हम इस उदाहरण में `optimi-lion` इस्तेमाल करेंगे।
- `mixed_precision` - सबसे प्रभावी ट्रेनिंग कॉन्फ़िगरेशन के लिए `bf16` रखने की सलाह है, या `no` (लेकिन इससे अधिक मेमोरी लगेगी और ट्रेनिंग धीमी होगी)।
- `gradient_checkpointing` - इसे बंद करने से सबसे तेज़ चलेगा, लेकिन बैच साइज़ सीमित हो जाएंगे। सबसे कम VRAM उपयोग के लिए इसे सक्षम करना ज़रूरी है।

HiDream के कुछ उन्नत विकल्प ट्रेनिंग के दौरान MoE auxiliary loss को शामिल करने के लिए सेट किए जा सकते हैं। MoE loss जोड़ने पर वैल्यू सामान्य से काफी अधिक हो जाती है।

- `hidream_use_load_balancing_loss` - लोड बैलेंसिंग लॉस सक्षम करने के लिए `true` सेट करें।
- `hidream_load_balancing_loss_weight` - यह auxiliary loss की तीव्रता है। `0.01` डिफ़ॉल्ट है, लेकिन आप इसे `0.1` या `0.2` तक बढ़ा सकते हैं।

इन विकल्पों का प्रभाव अभी ज्ञात नहीं है।

अंत में आपका config.json कुछ ऐसा दिखेगा:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
    "validation_torch_compile": "false",
    "validation_step_interval": 200,
    "validation_seed": 42,
    "validation_resolution": "1024x1024",
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_num_inference_steps": "20",
    "validation_guidance": 3.0,
    "validation_guidance_rescale": "0.0",
    "vae_batch_size": 1,
    "train_batch_size": 1,
    "tracker_run_name": "eval_loss_test1",
    "seed": 42,
    "resume_from_checkpoint": "latest",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "report_to": "tensorboard",
    "output_dir": "output/models-hidream",
    "optimizer": "optimi-lion",
    "num_train_epochs": 0,
    "num_eval_images": 1,
    "model_type": "lora",
    "model_family": "hidream",
    "offload_during_startup": true,
    "mixed_precision": "bf16",
    "minimum_image_size": 0,
    "max_train_steps": 10000,
    "max_grad_norm": 0.01,
    "lycoris_config": "config/lycoris_config.json",
    "lr_warmup_steps": 100,
    "lr_scheduler": "constant_with_warmup",
    "lora_type": "lycoris",
    "learning_rate": "4e-5",
    "gradient_checkpointing": "true",
    "grad_clip_method": "value",
    "eval_steps_interval": 100,
    "disable_benchmark": false,
    "data_backend_config": "config/hidream/multidatabackend.json",
    "checkpoints_total_limit": 5,
    "checkpoint_step_interval": 500,
    "caption_dropout_probability": 0.0,
    "base_model_precision": "int8-quanto",
    "text_encoder_3_precision": "int8-quanto",
    "text_encoder_4_precision": "int8-quanto",
    "aspect_bucket_rounding": 2
}
```
</details>

> ℹ️ मल्टी-GPU उपयोगकर्ता GPUs की संख्या कॉन्फ़िगर करने के लिए [यह डॉक्यूमेंट](../OPTIONS.md#environment-configuration-variables) देख सकते हैं।

> ℹ️ यह कॉन्फ़िगरेशन मेमोरी बचाने के लिए T5 (#3) और Llama (#4) टेक्स्ट एन्कोडर की प्रिसीजन को int8 सेट करता है। यदि आपके पास अधिक मेमोरी है तो इन विकल्पों को हटा दें या `no_change` सेट करें।

और एक सरल `config/lycoris_config.json` फ़ाइल - ध्यान दें कि अतिरिक्त ट्रेनिंग स्थिरता के लिए `FeedForward` हटाया जा सकता है।

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
    "algo": "lokr",
    "multiplier": 1.0,
    "linear_dim": 16384,
    "linear_alpha": 1,
    "full_matrix": true,
    "use_scalar": true,
    "factor": 16,
    "apply_preset": {
        "target_module": [
            "Attention",
        ],
        "module_algo_map": {
            "Attention": {
                "factor": 16
            },
        }
    }
}
```
</details>

`config/lycoris_config.json` में `"use_scalar": true` सेट करने या `config/config.json` में `"init_lokr_norm": 1e-4` सेट करने से ट्रेनिंग काफी तेज़ हो जाएगी। दोनों सक्षम करने से थोड़ा धीमा लगता है। ध्यान दें कि `init_lokr_norm` सेट करने से स्टेप 0 पर वैलिडेशन इमेजेस थोड़ी बदलेंगी।

`config/lycoris_config.json` में `FeedForward` मॉड्यूल जोड़ने से बहुत बड़े संख्या में पैरामीटर ट्रेन होंगे, जिसमें सभी experts भी शामिल हैं। experts को ट्रेन करना हालांकि काफी कठिन लगता है।

एक आसान विकल्प यह है कि experts के बाहर केवल feed forward पैरामीटर्स ट्रेन करें, इसके लिए नीचे दिया गया `config/lycoris_config.json` उपयोग करें।

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
    "algo": "lokr",
    "multiplier": 1.0,
    "linear_dim": 16384,
    "linear_alpha": 1,
    "full_matrix": true,
    "use_scalar": true,
    "factor": 16,
    "apply_preset": {
        "name_algo_map": {
            "double_stream_blocks.*.block.attn*": {
                "factor": 16
            },
            "double_stream_blocks.*.block.ff_t*": {
                "factor": 16
            },
            "double_stream_blocks.*.block.ff_i.shared_experts*": {
                "factor": 16
            },
            "single_stream_blocks.*.block.attn*": {
                "factor": 16
            },
            "single_stream_blocks.*.block.ff_i.shared_experts*": {
                "factor": 16
            }
        },
        "use_fnmatch": true
    }
}
```
</details>

### उन्नत प्रयोगात्मक फीचर्स

<details>
<summary>उन्नत प्रयोगात्मक विवरण दिखाएँ</summary>


SimpleTuner में ऐसे प्रयोगात्मक फीचर्स शामिल हैं जो ट्रेनिंग की स्थिरता और परफॉर्मेंस को काफी बेहतर कर सकते हैं।

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** ट्रेनिंग के दौरान मॉडल को अपने इनपुट खुद बनाने देकर exposure bias कम करता है और आउटपुट गुणवत्ता बढ़ाता है।

> ⚠️ ये फीचर्स ट्रेनिंग का कम्प्यूटेशनल ओवरहेड बढ़ाते हैं।

#### वैलिडेशन प्रॉम्प्ट्स

`config/config.json` के अंदर "primary validation prompt" होता है, जो आमतौर पर वही मुख्य instance_prompt होता है जिस पर आप अपने एकल subject या style को ट्रेन कर रहे होते हैं। इसके अलावा, एक JSON फ़ाइल बनाई जा सकती है जिसमें वैलिडेशन के दौरान चलाने के लिए अतिरिक्त प्रॉम्प्ट्स होते हैं।

उदाहरण कॉन्फ़िग फ़ाइल `config/user_prompt_library.json.example` में यह फ़ॉर्मेट है:

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

nicknames वैलिडेशन की फ़ाइल-नाम होते हैं, इसलिए उन्हें छोटा और आपके फ़ाइलसिस्टम के अनुकूल रखें।

ट्रेनर को इस प्रॉम्प्ट लाइब्रेरी की ओर इंगित करने के लिए इसे TRAINER_EXTRA_ARGS में जोड़ें, `config.json` के अंत में एक नई लाइन जोड़कर:
```json
  "--user_prompt_library": "config/user_prompt_library.json",
```

विविध प्रॉम्प्ट्स का सेट यह निर्धारित करने में मदद करेगा कि मॉडल ट्रेनिंग के दौरान collapsing तो नहीं कर रहा। इस उदाहरण में `<token>` को अपने subject नाम (instance_prompt) से बदलें।

```json
{
    "anime_<token>": "a breathtaking anime-style portrait of <token>, capturing her essence with vibrant colors and expressive features",
    "chef_<token>": "a high-quality, detailed photograph of <token> as a sous-chef, immersed in the art of culinary creation",
    "just_<token>": "a lifelike and intimate portrait of <token>, showcasing her unique personality and charm",
    "cinematic_<token>": "a cinematic, visually stunning photo of <token>, emphasizing her dramatic and captivating presence",
    "elegant_<token>": "an elegant and timeless portrait of <token>, exuding grace and sophistication",
    "adventurous_<token>": "a dynamic and adventurous photo of <token>, captured in an exciting, action-filled moment",
    "mysterious_<token>": "a mysterious and enigmatic portrait of <token>, shrouded in shadows and intrigue",
    "vintage_<token>": "a vintage-style portrait of <token>, evoking the charm and nostalgia of a bygone era",
    "artistic_<token>": "an artistic and abstract representation of <token>, blending creativity with visual storytelling",
    "futuristic_<token>": "a futuristic and cutting-edge portrayal of <token>, set against a backdrop of advanced technology",
    "woman": "a beautifully crafted portrait of a woman, highlighting her natural beauty and unique features",
    "man": "a powerful and striking portrait of a man, capturing his strength and character",
    "boy": "a playful and spirited portrait of a boy, capturing youthful energy and innocence",
    "girl": "a charming and vibrant portrait of a girl, emphasizing her bright personality and joy",
    "family": "a heartwarming and cohesive family portrait, showcasing the bonds and connections between loved ones"
}
```

> ℹ️ HiDream डिफ़ॉल्ट रूप से 128 टोकन तक जाता है और फिर ट्रंकट करता है।

#### CLIP स्कोर ट्रैकिंग

यदि आप मॉडल की परफॉर्मेंस को स्कोर करने के लिए evaluations सक्षम करना चाहते हैं, तो CLIP स्कोर को कॉन्फ़िगर और व्याख्यायित करने के लिए [यह डॉक्यूमेंट](../evaluation/CLIP_SCORES.md) देखें।

</details>

# स्थिर मूल्यांकन लॉस

यदि आप मॉडल की परफॉर्मेंस को स्कोर करने के लिए stable MSE loss उपयोग करना चाहते हैं, तो evaluation loss को कॉन्फ़िगर और व्याख्यायित करने के लिए [यह डॉक्यूमेंट](../evaluation/EVAL_LOSS.md) देखें।

#### वैलिडेशन प्रीव्यूज़

SimpleTuner Tiny AutoEncoder मॉडल्स का उपयोग करके जेनरेशन के दौरान इंटरमीडिएट वैलिडेशन प्रीव्यूज़ स्ट्रीम करना सपोर्ट करता है। इससे आप वेबहुक कॉलबैक के जरिए वैलिडेशन इमेजेस को स्टेप-बाय-स्टेप रियल-टाइम में बनते देख सकते हैं।

सक्षम करने के लिए:
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
- Webhook कॉन्फ़िगरेशन
- Validation सक्षम होना

`validation_preview_steps` को अधिक वैल्यू (जैसे 3 या 5) पर सेट करें ताकि Tiny AutoEncoder का ओवरहेड कम हो। `validation_num_inference_steps=20` और `validation_preview_steps=5` के साथ आपको स्टेप 5, 10, 15, और 20 पर प्रीव्यू इमेज मिलेंगी।

#### फ्लो शेड्यूल शिफ्टिंग

OmniGen, Sana, Flux, और SD3 जैसे flow-matching मॉडल्स में "shift" नाम की एक प्रॉपर्टी होती है जिससे हम timestep शेड्यूल के ट्रेन किए हिस्से को एक साधारण दशमलव मान से शिफ्ट कर सकते हैं।

`full` मॉडल 3.0 के मान के साथ ट्रेन हुआ है और `dev` ने 6.0 का उपयोग किया।

प्रैक्टिस में, इतना उच्च shift मान अक्सर दोनों मॉडलों को बिगाड़ देता है। 1.0 एक अच्छा प्रारंभिक मान है, लेकिन यह मॉडल को बहुत कम शिफ्ट कर सकता है, और 3.0 बहुत अधिक हो सकता है।

##### ऑटो-शिफ्ट

आमतौर पर सुझाया गया तरीका यह है कि हाल के कई कार्यों का अनुसरण करें और रेजोल्यूशन-डिपेंडेंट timestep shift सक्षम करें, `--flow_schedule_auto_shift`, जो बड़े इमेज के लिए उच्च shift और छोटे इमेज के लिए कम shift उपयोग करता है। यह स्थिर लेकिन संभवतः औसत ट्रेनिंग परिणाम देता है।

##### मैनुअल निर्दिष्ट करना

_Discord से General Awareness का उदाहरणों के लिए धन्यवाद_

`--flow_schedule_shift` का मान 0.1 (बहुत कम) उपयोग करने पर, केवल इमेज के सूक्ष्म विवरण प्रभावित होते हैं:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

`--flow_schedule_shift` का मान 4.0 (बहुत अधिक) उपयोग करने पर, मॉडल की बड़े कंपोज़िशनल फीचर्स और संभवतः रंग-स्थान प्रभावित हो जाते हैं:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)


#### डेटासेट पर विचार

अपने मॉडल को ट्रेन करने के लिए पर्याप्त बड़ा डेटासेट होना बहुत ज़रूरी है। डेटासेट आकार पर सीमाएँ हैं, और आपको यह सुनिश्चित करना होगा कि आपका डेटासेट प्रभावी ट्रेनिंग के लिए पर्याप्त बड़ा हो। ध्यान दें कि न्यूनतम डेटासेट आकार `train_batch_size * gradient_accumulation_steps` और `vae_batch_size` से अधिक होना चाहिए। यदि डेटासेट बहुत छोटा है तो वह उपयोग करने योग्य नहीं होगा।

> ℹ️ यदि इमेज बहुत कम हैं, तो आपको **no images detected in dataset** संदेश दिख सकता है - `repeats` मान बढ़ाने से यह सीमा पार हो जाएगी।

आपके पास जो डेटासेट है, उसके आधार पर आपको अपने डेटासेट डायरेक्टरी और डाटालोडर कॉन्फ़िगरेशन फ़ाइल को अलग तरीके से सेट करना होगा। इस उदाहरण में, हम [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) डेटासेट का उपयोग करेंगे।

`--data_backend_config` (`config/multidatabackend.json`) डॉक्यूमेंट में यह जोड़ें:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
[
  {
    "id": "pseudo-camera-10k-hidream",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/hidream/pseudo-camera-10k",
    "instance_data_dir": "datasets/pseudo-camera-10k",
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "discovery",
    "repeats": 0,
    "is_regularisation_data": true
  },
  {
    "id": "dreambooth-subject",
    "type": "local",
    "crop": false,
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/hidream/dreambooth-subject",
    "instance_data_dir": "datasets/dreambooth-subject",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/hidream",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> ℹ️ यदि आपके पास कैप्शन वाली `.txt` फ़ाइलें हैं तो `caption_strategy=textfile` उपयोग करें।
> `caption_strategy` विकल्प और आवश्यकताएँ [DATALOADER.md](../DATALOADER.md#caption_strategy) में देखें।

फिर, एक `datasets` डायरेक्टरी बनाएं:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

यह आपके `datasets/pseudo-camera-10k` डायरेक्टरी में लगभग 10k फ़ोटोग्राफ़ सैंपल डाउनलोड करेगा, जो आपके लिए अपने आप बन जाएगी।

आपकी Dreambooth इमेजेस `datasets/dreambooth-subject` डायरेक्टरी में जानी चाहिए।

#### WandB और Huggingface Hub में लॉगिन करें

ट्रेनिंग शुरू करने से पहले WandB और HF Hub में लॉगिन करना अच्छा होगा, खासकर यदि आप `--push_to_hub` और `--report_to=wandb` उपयोग कर रहे हैं।

यदि आप Git LFS रिपॉज़िटरी में मैन्युअली आइटम पुश करने वाले हैं, तो `git config --global credential.helper store` भी चलाएँ।

निम्न कमांड चलाएँ:

```bash
wandb login
```

और

```bash
huggingface-cli login
```

दोनों सेवाओं में लॉगिन के लिए निर्देशों का पालन करें।

### ट्रेनिंग रन चलाना

SimpleTuner डायरेक्टरी से, ट्रेनिंग शुरू करने के लिए आपके पास कई विकल्प हैं:

**विकल्प 1 (अनुशंसित - pip इंस्टॉल):**
```bash
pip install simpletuner[cuda]
simpletuner train
```

**विकल्प 2 (Git clone विधि):**
```bash
simpletuner train
```

**विकल्प 3 (Legacy विधि - अभी भी काम करती है):**
```bash
./train.sh
```

यह टेक्स्ट एम्बेड और VAE आउटपुट को डिस्क पर कैश करना शुरू करेगा।

अधिक जानकारी के लिए [dataloader](../DATALOADER.md) और [tutorial](../TUTORIAL.md) डॉक्यूमेंट देखें।

### इसके बाद LoKr पर inference चलाना

क्योंकि यह नया मॉडल है, उदाहरण को काम कराने के लिए कुछ एडजस्टमेंट की जरूरत होगी। यहाँ एक काम करने वाला उदाहरण है:

<details>
<summary>Python inference उदाहरण दिखाएँ</summary>

```py
import torch
from helpers.models.hidream.pipeline import HiDreamImagePipeline
from helpers.models.hidream.transformer import HiDreamImageTransformer2DModel
from lycoris import create_lycoris_from_weights
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM

llama_repo = "unsloth/Meta-Llama-3.1-8B-Instruct"
model_id = 'HiDream-ai/HiDream-I1-Dev'
adapter_repo_id = 'bghira/hidream5m-photo-1mp-Prodigy'
adapter_filename = 'pytorch_lora_weights.safetensors'

tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
    llama_repo,
)
text_encoder_4 = LlamaForCausalLM.from_pretrained(
    llama_repo,
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=torch.bfloat16,
)

def download_adapter(repo_id: str):
    import os
    from huggingface_hub import hf_hub_download
    adapter_filename = "pytorch_lora_weights.safetensors"
    cache_dir = os.environ.get('HF_PATH', os.path.expanduser('~/.cache/huggingface/hub/models'))
    cleaned_adapter_path = repo_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    path_to_adapter = os.path.join(cache_dir, cleaned_adapter_path)
    path_to_adapter_file = os.path.join(path_to_adapter, adapter_filename)
    os.makedirs(path_to_adapter, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id, filename=adapter_filename, local_dir=path_to_adapter
    )

    return path_to_adapter_file

adapter_file_path = download_adapter(repo_id=adapter_repo_id)
transformer = HiDreamImageTransformer2DModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, subfolder="transformer")
pipeline = HiDreamImagePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    tokenizer_4=tokenizer_4,
    text_encoder_4=text_encoder_4,
    transformer=transformer,
)
lora_scale = 1.0
wrapper, _ = create_lycoris_from_weights(lora_scale, adapter_file_path, pipeline.transformer)
wrapper.merge_to()

prompt = "Place your test prompt here."
negative_prompt = 'ugly, cropped, blurry, low-quality, mediocre average'

## Optional: quantise the model to save on vram.
## Note: The model was quantised during training, and so it is recommended to do the same during inference time.
from optimum.quanto import quantize, freeze, qint8
quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)

pipeline.to('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') # the pipeline is already in its target precision level
t5_embeds, llama_embeds, negative_t5_embeds, negative_llama_embeds, pooled_embeds, negative_pooled_embeds = pipeline.encode_prompt(
    prompt=prompt, prompt_2=prompt, prompt_3=prompt, prompt_4=prompt, num_images_per_prompt=1
)
# We'll nuke the text encoders to save memory.
pipeline.text_encoder.to("meta")
pipeline.text_encoder_2.to("meta")
pipeline.text_encoder_3.to("meta")
pipeline.text_encoder_4.to("meta")
model_output = pipeline(
    t5_prompt_embeds=t5_embeds,
    llama_prompt_embeds=llama_embeds,
    pooled_prompt_embeds=pooled_embeds,
    negative_t5_prompt_embeds=negative_t5_embeds,
    negative_llama_prompt_embeds=negative_llama_embeds,
    negative_pooled_prompt_embeds=negative_pooled_embeds,
    num_inference_steps=30,
    generator=torch.Generator(device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu').manual_seed(42),
    width=1024,
    height=1024,
    guidance_scale=3.2,
).images[0]

model_output.save("output.png", format="PNG")

```
</details>

## नोट्स और ट्रबलशूटिंग टिप्स

### न्यूनतम VRAM कॉन्फ़िग

सबसे कम VRAM वाला HiDream कॉन्फ़िग लगभग 20-22G है:

- OS: Ubuntu Linux 24
- GPU: एक सिंगल NVIDIA CUDA डिवाइस (10G, 12G)
- System memory: लगभग 50G सिस्टम मेमोरी (कम भी हो सकती है, ज्यादा भी)
- Base model precision:
  - Apple और AMD सिस्टम के लिए, `int8-quanto` (या `fp8-torchao`, `int8-torchao` सभी समान मेमोरी उपयोग प्रोफाइल फॉलो करते हैं)
    - `int4-quanto` भी काम करता है, लेकिन आपको कम एक्यूरेसी / खराब परिणाम मिल सकते हैं
  - NVIDIA सिस्टम के लिए, `nf4-bnb` के काम करने की रिपोर्ट है, लेकिन यह `int8-quanto` से धीमा होगा
- Optimiser: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolution: 1024px
- Batch size: 1, शून्य gradient accumulation steps
- DeepSpeed: निष्क्रिय / अनकॉन्फ़िगर
- PyTorch: 2.7+
- स्टार्टअप पर <=16G कार्ड्स में outOfMemory त्रुटि से बचने के लिए `--quantize_via=cpu` का उपयोग करें।
- `--gradient_checkpointing` सक्षम करें
- एक छोटा LoRA या Lycoris कॉन्फ़िगरेशन इस्तेमाल करें (जैसे LoRA rank 1 या Lokr factor 25)
- एनवायरनमेंट वेरिएबल `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` सेट करने से कई आस्पेक्ट रेशियो ट्रेन करने पर VRAM उपयोग कम रखने में मदद मिलती है।

**नोट**: VAE embeds और टेक्स्ट एन्कोडर आउटपुट का प्री-कैश अधिक मेमोरी ले सकता है और फिर भी OOM हो सकता है। VAE टाइलिंग और स्लाइसिंग डिफ़ॉल्ट रूप से सक्षम हैं। यदि OOM दिखे तो `offload_during_startup=true` सक्षम करने की कोशिश करें; अन्यथा शायद पर्याप्त मेमोरी नहीं है।

गति लगभग 3 iterations प्रति सेकंड थी, NVIDIA 4090 पर Pytorch 2.7 और CUDA 12.8 के साथ

### मास्क्ड लॉस

यदि आप किसी subject या style को ट्रेन कर रहे हैं और किसी एक को मास्क करना चाहते हैं, तो Dreambooth गाइड के [masked loss training](../DREAMBOOTH.md#masked-loss) सेक्शन देखें।

### क्वांटाइज़ेशन

हालाँकि `int8` गति/गुणवत्ता बनाम मेमोरी के लिए सर्वोत्तम विकल्प है, `nf4` और `int4` भी उपलब्ध हैं। `int4` HiDream के लिए अनुशंसित नहीं है, क्योंकि यह खराब परिणाम दे सकता है, लेकिन पर्याप्त ट्रेनिंग के बाद आपके पास एक ठीक-ठाक `int4` मॉडल हो सकता है।

### लर्निंग रेट्स

#### LoRA (--lora_type=standard)

- छोटे LoRA (rank-1 से rank-8) के लिए लगभग 4e-4 जैसे उच्च लर्निंग रेट बेहतर काम करते हैं
- बड़े LoRA (rank-64 से rank-256) के लिए लगभग 6e-5 जैसे कम लर्निंग रेट बेहतर काम करते हैं
- `lora_alpha` को `lora_rank` से अलग सेट करना Diffusers की सीमाओं के कारण समर्थित नहीं है, जब तक आप बाद में इंफरेंस टूल्स में यह समझकर न करें।
  - बाद में इंफरेंस में इसे कैसे उपयोग करें यह इस डॉक्यूमेंट के दायरे से बाहर है, लेकिन `lora_alpha` को 1.0 पर सेट करने से सभी LoRA ranks के लिए लर्निंग रेट समान रखा जा सकता है।

#### LoKr (--lora_type=lycoris)

- LoKr के लिए माइल्ड लर्निंग रेट बेहतर हैं (`1e-4` AdamW के साथ, `2e-5` Lion के साथ)
- अन्य algo को और एक्सप्लोरेशन की जरूरत है।
- Prodigy LoRA या LoKr के लिए अच्छा विकल्प लगता है, लेकिन यह आवश्यक लर्निंग रेट को अधिक अनुमानित कर सकता है और त्वचा को स्मूद कर सकता है।

### इमेज आर्टिफैक्ट्स

HiDream की इमेज आर्टिफैक्ट्स पर प्रतिक्रिया ज्ञात नहीं है, हालांकि यह Flux VAE उपयोग करता है और इसमें समान fine-details सीमाएँ हैं।

सबसे आम समस्या बहुत अधिक लर्निंग रेट और/या बहुत कम बैच साइज़ का उपयोग है। इससे मॉडल स्मूद स्किन, ब्लर, और पिक्सलेशन जैसे आर्टिफैक्ट्स वाली इमेज बना सकता है।

### आस्पेक्ट बकेटिंग

शुरुआत में मॉडल आस्पेक्ट बकेट्स पर बहुत अच्छा रिस्पॉन्स नहीं दे रहा था, लेकिन कम्युनिटी ने इम्प्लीमेंटेशन बेहतर कर दिया है।

### मल्टीपल-रेजोल्यूशन ट्रेनिंग

मॉडल को शुरुआत में 512px जैसे कम रेजोल्यूशन पर ट्रेन करके ट्रेनिंग तेज़ की जा सकती है, लेकिन यह स्पष्ट नहीं है कि मॉडल उच्च रेजोल्यूशन पर कितना अच्छा जनरलाइज़ करेगा। पहले 512px और फिर 1024px पर क्रमिक ट्रेनिंग शायद सबसे अच्छा तरीका है।

1024px से अलग रेजोल्यूशन पर ट्रेन करते समय `--flow_schedule_auto_shift` सक्षम करना अच्छा है। कम रेजोल्यूशन कम VRAM उपयोग करता है, जिससे बड़े बैच साइज़ संभव होते हैं।

### फुल-रैंक ट्यूनिंग

HiDream के साथ DeepSpeed बहुत अधिक सिस्टम मेमोरी उपयोग करेगा, लेकिन फुल ट्यूनिंग एक बहुत बड़े सिस्टम पर ठीक से काम करती है।

फुल-रैंक ट्यूनिंग की जगह Lycoris LoKr की सिफारिश की जाती है, क्योंकि यह अधिक स्थिर और कम मेमोरी उपयोग वाला है।

PEFT LoRA सरल स्टाइल्स के लिए उपयोगी है, लेकिन fine details बनाए रखना कठिन होता है।
