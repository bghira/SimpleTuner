# Flux[dev] / Flux[schnell] क्विकस्टार्ट

![image](https://github.com/user-attachments/assets/6409d790-3bb4-457c-a4b4-a51a45fc91d1)

इस उदाहरण में हम एक Flux.1 Krea LoRA ट्रेन करेंगे।

## हार्डवेयर आवश्यकताएँ

Flux को GPU मेमोरी के अलावा बहुत अधिक **system RAM** चाहिए। केवल startup पर मॉडल को quantize करने के लिए भी लगभग 50GB system memory चाहिए। यदि यह अत्यधिक समय ले रहा हो, तो अपने हार्डवेयर की क्षमता और आवश्यक बदलावों का आकलन करें।

जब आप rank-16 LoRA के हर component (MLP, projections, multimodal blocks) ट्रेन करते हैं, तो उपयोग लगभग इस प्रकार होता है:

- base model को quantize न करने पर 30G+ VRAM
- int8 + bf16 base/LoRA weights पर quantize करने पर 18G+ VRAM
- int4 + bf16 base/LoRA weights पर quantize करने पर 13G+ VRAM
- NF4 + bf16 base/LoRA weights पर quantize करने पर 9G+ VRAM
- int2 + bf16 base/LoRA weights पर quantize करने पर 9G+ VRAM

आपको चाहिए:

- **absolute minimum**: एक **3080 10G**
- **realistic minimum**: एक 3090 या V100 GPU
- **ideally**: कई 4090, A6000, L40S, या बेहतर

सौभाग्य से, ये [LambdaLabs](https://lambdalabs.com) जैसे providers पर आसानी से उपलब्ध हैं, जो कम दरों और multi-node training के लिए लोकलाइज्ड क्लस्टर प्रदान करते हैं।

**अन्य मॉडलों के विपरीत, Apple GPUs फिलहाल Flux ट्रेनिंग के लिए काम नहीं करते।**


## प्रीरिक्विज़िट्स

सुनिश्चित करें कि Python इंस्टॉल है; SimpleTuner 3.10 से 3.12 तक अच्छा काम करता है।

आप इसे चलाकर जांच सकते हैं:

```bash
python --version
```

यदि Ubuntu पर Python 3.12 इंस्टॉल नहीं है, तो यह आज़माएँ:

```bash
apt -y install python3.12 python3.12-venv
```

### कंटेनर इमेज निर्भरताएँ

Vast, RunPod, और TensorDock (और अन्य) के लिए, CUDA 12.2-12.8 इमेज पर CUDA एक्सटेंशन कम्पाइल करने हेतु यह काम करेगा:

```bash
apt -y install nvidia-cuda-toolkit
```

## इंस्टॉलेशन

pip के जरिए SimpleTuner इंस्टॉल करें:

```bash
pip install simpletuner[cuda]
```

मैनुअल इंस्टॉलेशन या डेवलपमेंट सेटअप के लिए [इंस्टॉलेशन डॉक्यूमेंटेशन](../INSTALL.md) देखें।

### AMD ROCm फॉलो‑अप स्टेप्स

AMD MI300X के लिए उपयोग योग्य होने हेतु ये स्टेप्स आवश्यक हैं:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

## एनवायरनमेंट सेटअप करना

### वेब इंटरफेस विधि

SimpleTuner WebUI सेटअप को काफी सरल बनाता है। सर्वर चलाने के लिए:

```bash
simpletuner server
```

यह डिफ़ॉल्ट रूप से पोर्ट 8001 पर webserver बनाएगा, जिसे आप http://localhost:8001 पर खोल सकते हैं।

### मैनुअल / command‑line विधि

Command-line tools के जरिए SimpleTuner चलाने के लिए आपको एक configuration फ़ाइल, dataset और model directories, और एक dataloader configuration फ़ाइल सेट करनी होगी।

#### Configuration फ़ाइल

एक प्रयोगात्मक स्क्रिप्ट, `configure.py`, इंटरएक्टिव स्टेप-बाय-स्टेप कॉन्फ़िगरेशन के जरिए इस सेक्शन को पूरी तरह स्किप करने दे सकती है। इसमें कुछ सेफ्टी फीचर्स हैं जो सामान्य गलतियों से बचाते हैं।

**नोट:** यह आपके dataloader को कॉन्फ़िगर नहीं करता। यह आपको बाद में मैन्युअली करना होगा।

इसे चलाने के लिए:

```bash
simpletuner configure
```

> ⚠️ जिन देशों में Hugging Face Hub आसानी से उपलब्ध नहीं है, वहाँ के उपयोगकर्ताओं को अपने `~/.bashrc` या `~/.zshrc` में `HF_ENDPOINT=https://hf-mirror.com` जोड़ना चाहिए, आपके सिस्टम का `$SHELL` जो भी उपयोग कर रहा हो उसके अनुसार।

यदि आप मैनुअली कॉन्फ़िगर करना चाहते हैं:

`config/config.json.example` को `config/config.json` में कॉपी करें:

```bash
cp config/config.json.example config/config.json
```

वहाँ आपको संभवतः निम्न वेरिएबल्स बदलने होंगे:

- `model_type` - इसे `lora` सेट करें।
- `model_family` - इसे `flux` सेट करें।
- `model_flavour` - डिफ़ॉल्ट `krea` है, लेकिन मूल FLUX.1-Dev रिलीज़ ट्रेन करने के लिए `dev` सेट कर सकते हैं।
  - `krea` - डिफ़ॉल्ट FLUX.1-Krea [dev] मॉडल, Krea 1 का open-weights variant, BFL और Krea.ai का proprietary collaboration मॉडल
  - `dev` - Dev model flavour, पिछला डिफ़ॉल्ट
  - `schnell` - Schnell model flavour; quickstart fast noise schedule और assistant LoRA stack अपने आप सेट कर देता है
  - `kontext` - Kontext training (विशिष्ट निर्देशों के लिए [यह गाइड](../quickstart/FLUX_KONTEXT.md) देखें)
  - `fluxbooru` - FLUX.1-Dev आधारित de-distilled (CFG आवश्यक) मॉडल [FluxBooru](https://hf.co/terminusresearch/fluxbooru-v0.3), terminus research group द्वारा बनाया गया
  - `libreflux` - FLUX.1-Schnell आधारित de-distilled मॉडल जिसे T5 text encoder इनपुट्स पर attention masking चाहिए
- `offload_during_startup` - यदि VAE encodes के दौरान memory खत्म हो जाए तो `true` सेट करें।
- `pretrained_model_name_or_path` - इसे `black-forest-labs/FLUX.1-dev` सेट करें।
- `pretrained_vae_model_name_or_path` - इसे `black-forest-labs/FLUX.1-dev` सेट करें।
  - ध्यान दें कि इस मॉडल को डाउनलोड करने के लिए आपको Huggingface में लॉगिन और access grant चाहिए होगा। इस ट्यूटोरियल में हम लॉगिन पर बाद में चर्चा करेंगे।
- `output_dir` - इसे उस डायरेक्टरी पर सेट करें जहाँ आप checkpoints और validation images रखना चाहते हैं। पूर्ण path उपयोग करना बेहतर है।
- `train_batch_size` - इसे 1 रखें, खासकर यदि डेटासेट बहुत छोटा है।
- `validation_resolution` - Flux एक 1024px मॉडल है, इसलिए इसे `1024x1024` सेट कर सकते हैं।
  - Flux multi-aspect buckets पर fine-tuned है; अन्य resolutions को commas से अलग करके दे सकते हैं: `1024x1024,1280x768,2048x2048`
- `validation_guidance` - Flux के लिए inference में जो value आप सामान्यतः उपयोग करते हैं वही रखें।
- `validation_guidance_real` - Flux inference के लिए CFG उपयोग करने हेतु >1.0 रखें। validations धीमे होंगे, लेकिन परिणाम बेहतर होंगे। खाली `VALIDATION_NEGATIVE_PROMPT` के साथ सबसे अच्छा चलता है।
- `validation_num_inference_steps` - समय बचाने और अच्छी गुणवत्ता देखने के लिए ~20 रखें। Flux बहुत विविध नहीं है, और अधिक steps शायद समय ही बर्बाद करेंगे।
- `--lora_rank=4` यदि आप LoRA का आकार काफी कम करना चाहते हैं, इससे VRAM उपयोग कम होगा।
- Schnell LoRA runs fast schedule को quickstart defaults के जरिए अपने आप उपयोग करती हैं; अतिरिक्त flags की जरूरत नहीं।

- `gradient_accumulation_steps` - पहले guidance थी कि bf16 ट्रेनिंग में इसे avoid करें क्योंकि यह मॉडल degrade कर सकता है। आगे के परीक्षण में Flux के लिए यह आवश्यक रूप से सही नहीं पाया गया।
  - यह विकल्प कई steps में update steps accumulate करता है। इससे ट्रेनिंग runtime linear रूप से बढ़ेगी, जैसे 2 रखने पर ट्रेनिंग आधी गति से चलेगी और दोगुना समय लेगी।
- `optimizer` - शुरुआती users के लिए adamw_bf16 पर टिके रहना बेहतर है, हालांकि optimi-lion और optimi-stableadamw भी अच्छे विकल्प हैं।
- `mixed_precision` - शुरुआती users इसे `bf16` पर रखें
- `gradient_checkpointing` - लगभग हर स्थिति में हर डिवाइस पर true रखें
- `gradient_checkpointing_interval` - बड़े GPUs पर इसे 2 या अधिक पर सेट कर सकते हैं ताकि हर _n_ blocks पर ही checkpoint हो। 2 का मतलब आधे blocks, 3 का मतलब एक-तिहाई blocks।

### उन्नत प्रयोगात्मक फीचर्स

<details>
<summary>उन्नत प्रयोगात्मक विवरण दिखाएँ</summary>


SimpleTuner में ऐसे प्रयोगात्मक फीचर्स शामिल हैं जो ट्रेनिंग की स्थिरता और परफॉर्मेंस को काफी बेहतर कर सकते हैं।

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** exposure bias कम करता है और आउटपुट गुणवत्ता बेहतर करता है क्योंकि ट्रेनिंग के दौरान मॉडल अपने इनपुट्स स्वयं जनरेट करता है।

> ⚠️ ये फीचर्स ट्रेनिंग का कम्प्यूटेशनल ओवरहेड बढ़ाते हैं।

</details>

### Memory offloading (वैकल्पिक)

Flux diffusers v0.33+ के जरिए grouped module offloading सपोर्ट करता है। यह transformer weights की वजह से VRAM bottleneck होने पर VRAM दबाव को काफी कम करता है। आप इसे `TRAINER_EXTRA_ARGS` (या WebUI Hardware page) में यह flags जोड़कर सक्षम कर सकते हैं:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- `--group_offload_use_stream` केवल CUDA devices पर प्रभावी है; SimpleTuner ROCm, MPS और CPU backends पर streams स्वतः बंद कर देता है।
- इसे `--enable_model_cpu_offload` के साथ **साथ में न** उपयोग करें — दोनों strategies परस्पर असंगत हैं।
- `--group_offload_to_disk_path` उपयोग करते समय तेज़ local SSD/NVMe target चुनें।

#### Validation prompts

`config/config.json` के अंदर "primary validation prompt" होता है, जो आमतौर पर वही मुख्य instance_prompt होता है जिस पर आप अपने single subject या style को ट्रेन कर रहे होते हैं। इसके अलावा, एक JSON फ़ाइल बनाई जा सकती है जिसमें validations के दौरान चलाने के लिए अतिरिक्त prompts होते हैं।

उदाहरण config फ़ाइल `config/user_prompt_library.json.example` में यह फ़ॉर्मेट है:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```
</details>

nicknames validation के फ़ाइलनाम होते हैं, इसलिए उन्हें छोटा और आपके फ़ाइलसिस्टम के अनुकूल रखें।

Trainer को इस prompt library की ओर पॉइंट करने के लिए इसे TRAINER_EXTRA_ARGS में जोड़ें, `config.json` के अंत में एक नई लाइन जोड़कर:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
  "--user_prompt_library": "config/user_prompt_library.json",
```
</details>

विविध prompts का सेट यह निर्धारित करने में मदद करेगा कि मॉडल ट्रेनिंग के दौरान collapsing तो नहीं कर रहा। इस उदाहरण में `<token>` को अपने subject नाम (instance_prompt) से बदलें।

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

#### CLIP स्कोर ट्रैकिंग

यदि आप मॉडल की परफॉर्मेंस को स्कोर करने के लिए evaluations सक्षम करना चाहते हैं, तो CLIP स्कोर को कॉन्फ़िगर और व्याख्यायित करने के लिए [यह डॉक्यूमेंट](../evaluation/CLIP_SCORES.md) देखें।

# Stable evaluation loss

यदि आप मॉडल की परफॉर्मेंस को स्कोर करने के लिए stable MSE loss उपयोग करना चाहते हैं, तो evaluation loss को कॉन्फ़िगर और व्याख्यायित करने के लिए [यह डॉक्यूमेंट](../evaluation/EVAL_LOSS.md) देखें।

#### Validation previews

SimpleTuner Tiny AutoEncoder मॉडल्स का उपयोग करके जेनरेशन के दौरान इंटरमीडिएट validation previews स्ट्रीम करना सपोर्ट करता है। इससे आप webhook callbacks के जरिए validation images को step-by-step real-time में बनते देख सकते हैं।

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

`validation_preview_steps` को अधिक मान (जैसे 3 या 5) पर सेट करें ताकि Tiny AutoEncoder का ओवरहेड कम हो। `validation_num_inference_steps=20` और `validation_preview_steps=5` के साथ आपको steps 5, 10, 15, और 20 पर preview images मिलेंगी।

#### Flux time schedule shifting

Flow-matching मॉडल्स जैसे OmniGen, Sana, Flux, और SD3 में "shift" नाम की property होती है जिससे हम timestep schedule के ट्रेन किए हिस्से को एक साधारण दशमलव मान से shift कर सकते हैं।

##### Defaults

`full` मॉडल 3.0 के मान के साथ ट्रेन हुआ है और `dev` ने 6.0 का उपयोग किया।

प्रैक्टिस में, इतना उच्च shift मान अक्सर दोनों मॉडलों को बिगाड़ देता है। 1.0 एक अच्छा शुरुआती मान है, लेकिन यह मॉडल को बहुत कम shift कर सकता है, और 3.0 बहुत अधिक हो सकता है।

##### Auto-shift

आमतौर पर सुझाया गया तरीका यह है कि हाल के कई कार्यों का अनुसरण करें और resolution-dependent timestep shift सक्षम करें, `--flow_schedule_auto_shift`, जो बड़े इमेज के लिए उच्च shift और छोटे इमेज के लिए कम shift उपयोग करता है। यह स्थिर लेकिन संभवतः औसत ट्रेनिंग परिणाम देता है।

##### Manual specification

_Discord से General Awareness का उदाहरणों के लिए धन्यवाद_

`--flow_schedule_shift` का मान 0.1 (बहुत कम) उपयोग करने पर, केवल इमेज के सूक्ष्म विवरण प्रभावित होते हैं:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

`--flow_schedule_shift` का मान 4.0 (बहुत अधिक) उपयोग करने पर, मॉडल की बड़े compositional फीचर्स और संभवतः रंग-स्थान प्रभावित हो जाते हैं:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)


#### Quantised model training

यदि आप base model को quantize करना चाहते हैं, तो कई विकल्प हैं:

- `int8-quanto` (सामान्यतः सर्वोत्तम)
- `int4-quanto` (कम मेमोरी, गुणवत्ता में संभावित गिरावट)
- `int8-torchao` / `int4-torchao`
- `nf4-bnb` (NVIDIA पर अच्छा, लेकिन धीमा)

> यदि आप `int4` का उपयोग कर रहे हैं, तो लंबे समय तक ट्रेनिंग करने पर परिणाम बेहतर हो सकते हैं।

##### LoRA-specific settings (not LyCORIS)

- `--flux_lora_target`: LoRA targets सेट करें (`all`, `attention`, `mlp`, `tiny`, `nano`, आदि)
- `--flux_lora_init`: LoftQ या अन्य initialisation options का उपयोग
- `--flux_lora_module_type`: LoRA module प्रकार चुनें

> नोट: यदि आप LoftQ initialisation उपयोग करना चाहते हैं, तो base model के लिए Quanto quantization उपयोग नहीं कर सकते। यह संभवतः बेहतर/तेज़ convergence देता है, लेकिन केवल NVIDIA पर काम करता है और Bits n Bytes चाहिए; Quanto के साथ incompatible है।

#### Dataset considerations

> ⚠️ Flux के लिए training में image quality अन्य अधिकांश मॉडलों से अधिक महत्वपूर्ण है, क्योंकि यह आपकी इमेजेस के artifacts को _सबसे पहले_ absorb करता है, और फिर concept/subject सीखता है।

मॉडल ट्रेन करने के लिए पर्याप्त बड़ा डेटासेट होना बहुत जरूरी है। डेटासेट आकार पर सीमाएँ हैं, और आपको यह सुनिश्चित करना होगा कि आपका डेटासेट प्रभावी ट्रेनिंग के लिए पर्याप्त बड़ा हो। ध्यान दें कि न्यूनतम डेटासेट आकार `train_batch_size * gradient_accumulation_steps` के साथ-साथ `vae_batch_size` से अधिक होना चाहिए। यदि डेटासेट बहुत छोटा है तो वह उपयोग करने योग्य नहीं होगा।

> ℹ️ यदि इमेज बहुत कम हैं, तो आपको **no images detected in dataset** संदेश दिख सकता है - `repeats` मान बढ़ाने से यह सीमा पार हो जाएगी।

आपके पास जो डेटासेट है, उसके आधार पर आपको डेटासेट डायरेक्टरी और dataloader configuration फ़ाइल अलग तरीके से सेट करनी होगी। इस उदाहरण में हम [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) डेटासेट का उपयोग करेंगे।

`--data_backend_config` (`config/multidatabackend.json`) डॉक्यूमेंट में यह जोड़ें:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
[
  {
    "id": "pseudo-camera-10k-flux",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/flux/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/flux/dreambooth-subject",
    "instance_data_dir": "datasets/dreambooth-subject",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "dreambooth-subject-512",
    "type": "local",
    "crop": false,
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/flux/dreambooth-subject-512",
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
    "cache_dir": "cache/text/flux",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> caption_strategy विकल्पों और आवश्यकताओं के लिए [DATALOADER.md](../DATALOADER.md#caption_strategy) देखें।

> ℹ️ 512px और 1024px datasets को एक साथ चलाना समर्थित है और Flux के लिए बेहतर convergence दे सकता है।

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

**विकल्प 1 (अनुशंसित - pip install):**

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

**नोट:** फिलहाल Flux के लिए multi-aspect buckets सही काम करते हैं या नहीं यह स्पष्ट नहीं है। `crop_style=random` और `crop_aspect=square` उपयोग करने की सलाह है।

## Multi-GPU कॉन्फ़िगरेशन

SimpleTuner WebUI के जरिए **automatic GPU detection** शामिल करता है। Onboarding के दौरान आप कॉन्फ़िगर करेंगे:

- **Auto Mode**: सभी detected GPUs को optimal settings के साथ अपने आप उपयोग करता है
- **Manual Mode**: विशिष्ट GPUs चुनें या custom process count सेट करें
- **Disabled Mode**: Single GPU training

WebUI आपके हार्डवेयर को detect करके `--num_processes` और `CUDA_VISIBLE_DEVICES` स्वचालित रूप से सेट करता है।

Manual कॉन्फ़िगरेशन या advanced सेटअप्स के लिए, installation guide में [Multi-GPU Training section](../INSTALL.md#multiple-gpu-training) देखें।

## Inference tips

### CFG-trained LoRAs (flux_guidance_value > 1)

ComfyUI में आपको Flux को AdaptiveGuider नाम के दूसरे node से गुज़ारना होगा। हमारी community के एक सदस्य ने modified node यहाँ दिया है:

(**external links**) [IdiotSandwichTheThird/ComfyUI-Adaptive-Guidan...](https://github.com/IdiotSandwichTheThird/ComfyUI-Adaptive-Guidance-with-disabled-init-steps) और उनका example workflow [यहाँ](https://github.com/IdiotSandwichTheThird/ComfyUI-Adaptive-Guidance-with-disabled-init-steps/blob/master/ExampleWorkflow.json)

### CFG-distilled LoRA (flux_guidance_scale == 1)

CFG-distilled LoRA का inference करना उतना ही आसान है जितना कि train किए गए value के आसपास एक कम guidance_scale उपयोग करना।

## Notes & troubleshooting tips

### न्यूनतम VRAM कॉन्फ़िग

फिलहाल, सबसे कम VRAM उपयोग (9090M) इस कॉन्फ़िग के साथ प्राप्त किया जा सकता है:

- OS: Ubuntu Linux 24
- GPU: एक सिंगल NVIDIA CUDA डिवाइस (10G, 12G)
- System memory: लगभग 50G सिस्टम मेमोरी
- Base model precision: `nf4-bnb`
- Optimiser: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolution: 512px
  - 1024px के लिए >= 12G VRAM चाहिए
- Batch size: 1, zero gradient accumulation steps
- DeepSpeed: disabled / unconfigured
- PyTorch: 2.6 Nightly (Sept 29th build)
- <=16G कार्ड्स पर startup के दौरान outOfMemory error से बचने के लिए `--quantize_via=cpu` उपयोग करें।
- `--attention_mechanism=sageattention` से VRAM 0.1GB कम और training validation image generation speed बेहतर होती है।
- `--gradient_checkpointing` सक्षम करना जरूरी है, वरना OOM से बच नहीं पाएँगे

**नोट**: VAE embeds और text encoder outputs का pre-caching अधिक मेमोरी ले सकता है और फिर भी OOM हो सकता है। यदि ऐसा हो, तो text encoder quantization और VAE tiling को `--vae_enable_tiling=true` से सक्षम करें। Startup पर और मेमोरी बचाने के लिए `--offload_during_startup=true` उपयोग करें।

स्पीड लगभग 1.4 iterations प्रति सेकंड थी, 4090 पर।

### SageAttention

`--attention_mechanism=sageattention` उपयोग करने पर validation समय में inference तेज़ हो सकता है।

**नोट**: यह हर मॉडल कॉन्फ़िगरेशन के साथ संगत नहीं है, लेकिन आज़माने लायक है।

### NF4-quantised training

सरल शब्दों में, NF4 मॉडल का 4bit-_ish_ representation है, इसलिए training stability को लेकर गंभीर चुनौतियाँ हैं।

शुरुआती टेस्ट में यह पाया गया:

- Lion optimiser मॉडल collapse कर सकता है लेकिन सबसे कम VRAM उपयोग करता है; AdamW variants मॉडल को स्थिर रखते हैं; bnb-adamw8bit, adamw_bf16 अच्छे विकल्प हैं
  - AdEMAMix अच्छा नहीं रहा, लेकिन सेटिंग्स की पूरी जाँच नहीं हुई
- `--max_grad_norm=0.01` बड़े बदलावों को रोककर मॉडल टूटने से बचाने में मदद करता है
- NF4, AdamW8bit, और higher batch size stability समस्याओं को कम करने में मदद करते हैं, लेकिन ट्रेनिंग समय या VRAM उपयोग बढ़ता है
- Resolution को 512px से 1024px करने पर ट्रेनिंग धीमी हो जाती है, उदाहरण के लिए 1.4 sec/step से 3.5 sec/step (batch size 1, 4090)
- जो चीज़ें int8 या bf16 पर ट्रेन करना कठिन है, NF4 पर और भी कठिन होती हैं
- यह SageAttention जैसे विकल्पों के साथ कम संगत है

NF4, torch.compile के साथ काम नहीं करता, इसलिए स्पीड वही मिलेगी जो मिलेगी।

यदि VRAM कोई समस्या नहीं है (जैसे 48G या अधिक) तो int8 + torch.compile सबसे तेज़ विकल्प है।

### Masked loss

यदि आप किसी subject या style को ट्रेन कर रहे हैं और किसी एक को mask करना चाहते हैं, तो Dreambooth गाइड के [masked loss training](../DREAMBOOTH.md#masked-loss) सेक्शन देखें।

### TREAD training

> ⚠️ **Experimental**: TREAD नई feature है। यह काम करती है, लेकिन optimal configurations अभी खोजे जा रहे हैं।

[TREAD](../TREAD.md) (पेपर) का मतलब है **T**oken **R**outing for **E**fficient **A**rchitecture-agnostic **D**iffusion। यह tokens को transformer layers में बुद्धिमानी से route करके Flux training को तेज़ कर सकता है। गति-वृद्धि इस बात पर निर्भर है कि आप कितने tokens drop करते हैं।

#### Quick setup

अपने `config.json` में यह जोड़ें:

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 2,
        "end_layer_idx": -2
      }
    ]
  }
}
```
</details>

यह कॉन्फ़िगरेशन करेगा:

- layers 2 से दूसरे‑आख़िरी तक केवल 50% image tokens रखेगा
- text tokens कभी drop नहीं होते
- ~25% training speedup, न्यूनतम quality impact के साथ

#### Key points

- **Limited architecture support** - TREAD केवल Flux और Wan मॉडलों के लिए इम्प्लीमेंटेड है
- **Best at high resolutions** - 1024x1024+ पर सबसे बड़ा speedup, क्योंकि attention की O(n²) complexity है
- **Compatible with masked loss** - masked regions स्वतः preserve रहते हैं (लेकिन speedup कम हो जाता है)
- **Works with quantization** - int8/int4/NF4 training के साथ जोड़ा जा सकता है
- **Expect initial loss spike** - LoRA/LoKr ट्रेनिंग शुरू करते समय loss ऊँचा होगा लेकिन जल्दी ठीक हो जाएगा

#### Tuning tips

- **Conservative (quality-focused)**: `selection_ratio` 0.3-0.5 रखें
- **Aggressive (speed-focused)**: `selection_ratio` 0.6-0.8 रखें
- **Avoid early/late layers**: layer 0-1 या अंतिम layer में routing न करें
- **For LoRA training**: हल्की धीमी गति दिख सकती है - अलग configs आज़माएँ
- **Higher resolution = बेहतर speedup**: 1024px और ऊपर पर सबसे लाभकारी

#### Known behavior

- ज्यादा tokens drop (ऊँचा `selection_ratio`) करने पर training तेज़, लेकिन प्रारंभिक loss ऊँचा
- LoRA/LoKr ट्रेनिंग में प्रारंभिक loss spike होता है जो नेटवर्क adaption के साथ जल्दी ठीक होता है
- कुछ LoRA configs थोड़ा धीमे ट्रेन हो सकते हैं - optimal configs अभी खोजे जा रहे हैं
- RoPE (rotary position embedding) implementation functional है लेकिन शायद 100% सही नहीं

विस्तृत कॉन्फ़िगरेशन विकल्पों और troubleshooting के लिए [full TREAD documentation](../TREAD.md) देखें।

### Classifier-free guidance

#### समस्या

Dev मॉडल आउट-ऑफ-द-बॉक्स guidance-distilled आता है, यानी यह teacher मॉडल आउटपुट्स के लिए बहुत सीधा trajectory बनाता है। यह एक guidance vector के जरिए होता है जिसे ट्रेनिंग और inference में मॉडल को दिया जाता है—इस vector का मान तय करता है कि आपको किस प्रकार का LoRA मिलेगा:

#### समाधान

- 1.0 का मान (**डिफ़ॉल्ट**) Dev मॉडल की शुरुआती distillation को बनाए रखता है
  - यह सबसे compatible मोड है
  - Inference मूल मॉडल जितना ही तेज़ है
  - Flow-matching distillation मॉडल की creativity और output variability को कम करता है, जैसा मूल Flux Dev मॉडल में होता है (composition/look समान रहते हैं)
- ऊँचा मान (लगभग 3.5-4.5) CFG उद्देश्य को मॉडल में वापस लाता है
  - इसके लिए inference pipeline में CFG सपोर्ट होना चाहिए
  - Inference 50% धीमा और 0% VRAM वृद्धि **या** batched CFG inference में लगभग 20% धीमा और 20% VRAM वृद्धि
  - लेकिन यह प्रशिक्षण शैली creativity और आउटपुट variability बढ़ाती है, जो कुछ training कार्यों के लिए आवश्यक हो सकती है

हम de-distilled मॉडल में 1.0 के मान के साथ ट्रेनिंग जारी रखकर distillation को आंशिक रूप से वापस ला सकते हैं। यह पूरी तरह रिकवर नहीं होगा, लेकिन उपयोगी हो जाएगा।

#### Caveats

- अंतिम प्रभाव **या तो**:
  - Inference latency को 2x बढ़ाना जब हम unconditional output को sequentially दो forward pass से निकालते हैं
  - VRAM consumption को `num_images_per_prompt=2` और inference पर दो इमेजेस पाने जितना बढ़ाना, साथ में समान प्रतिशत धीमापन।
    - यह तरीका sequential computation की तुलना में अक्सर कम धीमा होता है, लेकिन VRAM उपयोग अधिकांश consumer hardware के लिए बहुत अधिक हो सकता है।
    - यह तरीका फिलहाल SimpleTuner में integrated नहीं है, लेकिन काम जारी है।
- ComfyUI या अन्य applications (जैसे AUTOMATIC1111) के inference workflows को "true" CFG सक्षम करने के लिए बदलना होगा, जो अभी out-of-the-box संभव नहीं भी हो सकता।

### Quantisation

- 16G कार्ड पर इस मॉडल को ट्रेन करने के लिए न्यूनतम 8bit quantisation जरूरी है
  - bfloat16/float16 में rank-1 LoRA ~30GB से अधिक मेमोरी लेता है
- मॉडल को 8bit में quantize करने से ट्रेनिंग को नुकसान नहीं होता
  - यह higher batch sizes संभव करता है और बेहतर परिणाम भी दे सकता है
  - full-precision training जैसा ही व्यवहार करता है - fp32 आपके मॉडल को bf16+int8 से बेहतर नहीं बनाता।
- **int8** नए NVIDIA हार्डवेयर (3090 या बेहतर) पर hardware acceleration और `torch.compile()` सपोर्ट करता है
- **nf4-bnb** VRAM आवश्यकता को 9GB तक लाता है, जो 10G कार्ड पर फिट हो जाता है (bfloat16 सपोर्ट के साथ)
- बाद में ComfyUI में LoRA लोड करते समय, **आपको वही base model precision उपयोग करनी होगी जिस पर आपने LoRA ट्रेन किया था।**
- **int4** custom bf16 kernels पर निर्भर है, और यदि आपका कार्ड bfloat16 सपोर्ट नहीं करता तो काम नहीं करेगा

### Crashing

- यदि text encoders unload होने के बाद SIGKILL मिलता है, तो इसका मतलब है कि आपके पास Flux quantize करने के लिए पर्याप्त system memory नहीं है।
  - `--base_model_precision=bf16` से लोड करने की कोशिश करें, लेकिन यदि यह काम नहीं करता तो आपको अधिक मेमोरी चाहिए होगी।
  - GPU का उपयोग करने के लिए `--quantize_via=accelerator` आज़माएँ

### Schnell

- यदि आप Dev पर LyCORIS LoKr ट्रेन करते हैं, तो यह सामान्यतः Schnell पर सिर्फ 4 steps में बहुत अच्छा काम करता है।
  - सीधे Schnell ट्रेनिंग को थोड़ा और समय चाहिए - फिलहाल परिणाम अच्छे नहीं लगते

> ℹ️ Dev और Schnell को किसी भी तरह merge करने पर Dev का license लागू हो जाता है और मॉडल non-commercial हो जाता है। अधिकांश users के लिए यह मायने नहीं रखेगा, लेकिन ध्यान देने योग्य है।

### Learning rates

#### LoRA (--lora_type=standard)

- बड़े datasets पर LoRA का प्रदर्शन LoKr की तुलना में कुल मिलाकर खराब होता है
- रिपोर्ट है कि Flux LoRA, SD 1.5 LoRAs जैसा ट्रेन करता है
- हालांकि, 12B जैसा बड़ा मॉडल empirically **कम learning rates** के साथ बेहतर चलता है।
  - 1e-3 पर LoRA मॉडल को "भून" सकता है; 1e-5 पर लगभग कुछ नहीं होता।
- 12B मॉडल पर 64 से 128 तक के ranks सामान्य कठिनाइयों के कारण undesirable हो सकते हैं।
  - पहले छोटे नेटवर्क (rank-1, rank-4) से शुरू करें और धीरे-धीरे बढ़ाएँ — वे तेज़ ट्रेन होंगे और शायद आपकी जरूरत पूरी कर दें।
  - यदि आपके concept को मॉडल में ट्रेन करना अत्यधिक कठिन लग रहा है, तो आपको उच्च rank और अधिक regularisation डेटा चाहिए हो सकता है।
- PixArt और SD3 जैसे अन्य diffusion transformer मॉडल्स `--max_grad_norm` से काफी लाभ लेते हैं और SimpleTuner Flux के लिए डिफ़ॉल्ट रूप से काफी high value रखता है।
  - कम मान मॉडल को जल्दी टूटने से बचाते हैं, लेकिन नए concepts सीखना कठिन भी बना सकते हैं। मॉडल अटक सकता है और सुधार नहीं करेगा।

#### LoKr (--lora_type=lycoris)

- LoKr के लिए higher learning rates बेहतर हैं (`1e-3` AdamW के साथ, `2e-4` Lion के साथ)
- अन्य algo को और exploration चाहिए।
- ऐसे datasets पर `is_regularisation_data` सेट करने से bleed रोकने/संरक्षित करने में मदद मिल सकती है और अंतिम मॉडल की गुणवत्ता सुधार सकती है।
  - यह "prior loss preservation" से अलग व्यवहार करता है, जो training batch sizes को दोगुना करने के लिए जाना जाता है और परिणाम में बहुत सुधार नहीं करता
  - SimpleTuner का regularisation data implementation base model को संरक्षित करने का एक प्रभावी तरीका देता है

### Image artifacts

Flux तुरंत खराब image artifacts absorb कर लेता है। यही वास्तविकता है - अंत में सिर्फ high quality data पर एक अंतिम training run की जरूरत पड़ सकती है।

जब आप ये (और अन्य) चीज़ें करते हैं, तो samples में square grid artifacts **आ सकते हैं**:

- low quality data के साथ overtrain करना
- बहुत अधिक learning rate उपयोग करना
- overtraining (सामान्यतः), बहुत सारी images के साथ low-capacity नेटवर्क
- undertraining (भी), बहुत कम images के साथ high-capacity नेटवर्क
- अजीब aspect ratios या training data sizes उपयोग करना

### Aspect bucketing

- square crops पर बहुत लंबे समय तक training करने से इस मॉडल को बहुत नुकसान नहीं होगा। बेझिझक करें, यह अच्छा और reliable है।
- दूसरी ओर, अपने dataset के natural aspect buckets का उपयोग inference समय पर उन shapes को अधिक bias कर सकता है।
  - यह एक वांछनीय गुण हो सकता है, क्योंकि यह cinematic जैसे aspect-dependent styles को अन्य resolutions में बहुत अधिक bleed होने से रोकता है।
  - लेकिन यदि आप कई aspect buckets में समान रूप से परिणाम सुधारना चाहते हैं, तो `crop_aspect=random` के साथ प्रयोग करना पड़ सकता है, जिसका अपना downside है।
- अपने image directory dataset को कई बार परिभाषित कर dataset configurations को mix करने से बहुत अच्छे परिणाम और एक अच्छी तरह generalised model मिले हैं।

### Training custom fine-tuned Flux models

Hugging Face Hub पर कुछ fine-tuned Flux मॉडल्स (जैसे Dev2Pro) में पूर्ण directory structure नहीं होता, जिसके लिए इन specific options को सेट करना आवश्यक है।

`flux_guidance_value`, `validation_guidance_real` और `flux_attention_masked_training` विकल्पों को भी उसी तरह सेट करें जैसा creator ने किया था, यदि वह जानकारी उपलब्ध हो।

<details>
<summary>उदाहरण कॉन्फ़िग देखें</summary>

```json
{
    "model_family": "flux",
    "pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
    "pretrained_transformer_model_name_or_path": "ashen0209/Flux-Dev2Pro",
    "pretrained_vae_model_name_or_path": "black-forest-labs/FLUX.1-dev",
    "pretrained_transformer_subfolder": "none",
}
```
</details>
